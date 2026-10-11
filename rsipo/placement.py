"""Optional pinning of RSIPO's Ray work to a fixed set of nodes in a shared cluster.

RSIPO_RAY_NODES lists node IPs, comma-separated. When it is set, every placement group, actor and
environment worker the training job creates also asks for Ray's built-in ``node:<ip>`` resource of one
of those nodes, so nothing is scheduled anywhere else. When it is unset, the requests are unchanged.
"""
import os


def ray_nodes():
    return [ip.strip() for ip in os.environ.get("RSIPO_RAY_NODES", "").split(",") if ip.strip()]


def node_resource(index=0):
    """Custom-resource request placing work on the ``index``-th allowed node (cyclically); ``{}`` when unpinned."""
    nodes = ray_nodes()
    return {f"node:{nodes[index % len(nodes)]}": 0.001} if nodes else {}
