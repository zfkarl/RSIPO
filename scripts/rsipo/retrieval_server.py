"""SEED's e5 retrieval server with request batching.

Same index, corpus, encoder (e5-base-v2, fp16, mean pooling, max length 256), top-k, request schema, and response
format as examples/search/retriever/retrieval_server.py. Requests that arrive together are encoded and searched as one
batch, so exact flat search on CPU makes one pass over the 21M-vector index per batch instead of one per query.
"""
import argparse
import asyncio
import os
import queue
import sys
import threading
import time
from concurrent.futures import Future

import faiss
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../examples/search/retriever"))
import retrieval_server as seed  # noqa: E402

pending = queue.Queue()
app = FastAPI()
default_topk = 3


class ExactInnerProduct:
    """The flat index's exact inner-product top-k, computed with numpy's BLAS in database blocks.

    faiss-cpu's bundled BLAS runs this product about 14x slower than numpy's on the AMD EPYC hosts we used.
    The vectors are a zero-copy view of the loaded faiss index, so the search is the same flat search."""

    def __init__(self, index, block=1 << 21):
        n, d = index.ntotal, index.d
        self.xb = faiss.rev_swig_ptr(index.get_xb(), n * d).reshape(n, d)
        self.block = block

    def search(self, xq, k):
        best_scores = torch.full((len(xq), k), -float("inf"))
        best_ids = torch.zeros((len(xq), k), dtype=torch.int64)
        for start in range(0, len(self.xb), self.block):
            scores = torch.from_numpy(xq @ self.xb[start:start + self.block].T)
            top_scores, top_ids = torch.topk(scores, min(k, scores.shape[1]), dim=1)
            merged_scores = torch.cat([best_scores, top_scores], dim=1)
            merged_ids = torch.cat([best_ids, top_ids + start], dim=1)
            best_scores, order = torch.topk(merged_scores, k, dim=1)
            best_ids = torch.gather(merged_ids, 1, order)
        return best_scores.numpy(), best_ids.numpy()


def search_batch(retriever, queries, topks):
    k = max(topks)
    emb = retriever.encoder.encode(list(queries))
    scores, idxs = retriever.exact.search(emb, k) if hasattr(retriever, "exact") else retriever.index.search(emb, k=k)
    return [(seed.load_docs(retriever.corpus, row_idxs[:topk]), row_scores[:topk])
            for topk, row_scores, row_idxs in zip(topks, scores, idxs)]


def serve_batches(retriever, max_batch, max_wait):
    while True:
        batch = [pending.get()]
        deadline = time.monotonic() + max_wait
        while len(batch) < max_batch:
            left = deadline - time.monotonic()
            if left <= 0:
                break
            try:
                batch.append(pending.get(timeout=left))
            except queue.Empty:
                break
        try:
            results = search_batch(retriever, [q for q, _, _ in batch], [k for _, k, _ in batch])
            for (_, _, fut), result in zip(batch, results):
                fut.set_result(result)
        except Exception as exc:
            for _, _, fut in batch:
                if not fut.done():
                    fut.set_exception(exc)


@app.post("/retrieve")
async def retrieve_endpoint(request: seed.QueryRequest):
    fut = Future()
    pending.put((request.query, request.topk or default_topk, fut))
    docs, scores = await asyncio.wrap_future(fut)
    if request.return_scores:
        return {"result": [[{"document": doc, "score": float(score)} for doc, score in zip(docs, scores)]]}
    return {"result": [docs]}


def build_retriever(args):
    config = seed.Config(
        retrieval_method=args.retriever_name,
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        retrieval_topk=args.topk,
        faiss_gpu=args.faiss_gpu,
        retrieval_model_path=args.retriever_model,
        retrieval_pooling_method="mean",
        retrieval_query_max_length=256,
        retrieval_use_fp16=True,
        retrieval_batch_size=512,
    )
    retriever = seed.get_retriever(config)
    index = retriever.index
    if not args.faiss_gpu and isinstance(index, faiss.IndexFlat) and index.metric_type == faiss.METRIC_INNER_PRODUCT:
        retriever.exact = ExactInnerProduct(retriever.index)
    return retriever


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Batched e5 retriever with SEED's API.")
    p.add_argument("--index_path", required=True)
    p.add_argument("--corpus_path", required=True)
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--retriever_name", default="e5")
    p.add_argument("--retriever_model", required=True)
    p.add_argument("--faiss_gpu", action="store_true")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max_batch", type=int, default=256)
    p.add_argument("--max_wait_ms", type=float, default=20.0)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    default_topk = args.topk
    retriever = build_retriever(args)
    threading.Thread(target=serve_batches, args=(retriever, args.max_batch, args.max_wait_ms / 1000), daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
