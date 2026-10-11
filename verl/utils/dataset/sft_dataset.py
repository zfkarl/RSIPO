# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
SFT dataset
- We assume user pass a single parquet file.
- We load all the data into the memory.
Each parquet file contains
"""

import re
from typing import List, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask


class SFTDataset(Dataset):
    """
    This is an in-memory SFTDataset

    Arguments:
        config (OmegaConf): the data config
    """

    def __init__(self, parquet_files: Union[str, List[str]], tokenizer, config):
        prompt_key = config.get("prompt_key", "prompt")
        prompt_dict_keys = config.get("prompt_dict_keys", None)
        response_key = config.get("response_key", "response")
        response_dict_keys = config.get("response_dict_keys", None)
        max_length = config.get("max_length", 1024)
        truncation = config.get("truncation", "error")
        use_shm = config.get('use_shm', False)
        apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})

        assert truncation in ["error", "left", "right"]
        self.truncation = truncation
        self.use_shm = use_shm

        if not isinstance(parquet_files, List):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self.prompt_key = prompt_key if isinstance(prompt_key, (tuple, list)) else [prompt_key]
        self.response_key = response_key if isinstance(response_key, (tuple, list)) else [response_key]
        self.prompt_dict_keys = prompt_dict_keys if prompt_dict_keys else []
        self.response_dict_keys = response_dict_keys if response_dict_keys else []
        self.apply_chat_template_kwargs = dict(apply_chat_template_kwargs) if apply_chat_template_kwargs else {}

        self.max_length = max_length

        self._download()
        self._read_files_and_tokenize()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_to_local(parquet_file, verbose=True, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        def series_to_item(ls):
            import numpy
            import pandas

            while isinstance(ls, (pandas.core.series.Series, numpy.ndarray)) and len(ls) == 1:
                ls = ls[0]
            return ls

        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)
        self.prompts = self.dataframe[self.prompt_key]
        for key in self.prompt_dict_keys:
            # type(x): pandas.core.series.Series
            # type(x[0]): numpy.ndarray
            # type(x[0][0]): dict
            try:
                self.prompts = self.prompts.apply(lambda x: series_to_item(x)[key], axis=1)  # noqa: B023
            except Exception:
                print(f"self.prompts={self.prompts}")
                raise
        if isinstance(self.prompts, pd.DataFrame):
            self.prompts = self.prompts.squeeze("columns")
        self.prompts = self.prompts.tolist()
        self.responses = self.dataframe[self.response_key]
        for key in self.response_dict_keys:
            try:
                self.responses = self.responses.apply(lambda x: series_to_item(x)[key], axis=1)  # noqa: B023
            except Exception:
                print(f"self.responses={self.responses}")
                raise
        if isinstance(self.responses, pd.DataFrame):
            self.responses = self.responses.squeeze("columns")
        self.responses = self.responses.tolist()

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, item):
        tokenizer = self.tokenizer

        prompt = self.prompts[item]
        response = self.responses[item]

        # apply chat template
        prompt_chat = [{"role": "user", "content": prompt}]

        # string
        prompt_chat_str = tokenizer.apply_chat_template(
            prompt_chat,
            add_generation_prompt=True,
            tokenize=False,
            **self.apply_chat_template_kwargs,
        )
        response_chat_str = response + tokenizer.eos_token

        # tokenize
        prompt_ids_output = tokenizer(prompt_chat_str, return_tensors="pt", add_special_tokens=False)
        prompt_ids = prompt_ids_output["input_ids"][0]
        prompt_attention_mask = prompt_ids_output["attention_mask"][0]

        response_ids_output = tokenizer(response_chat_str, return_tensors="pt", add_special_tokens=False)
        response_ids = response_ids_output["input_ids"][0]
        response_attention_mask = response_ids_output["attention_mask"][0]

        prompt_length = prompt_ids.shape[0]
        response_length = response_ids.shape[0]

        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)

        # padding to max length
        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            padded_input_ids = torch.ones(size=(self.max_length - sequence_length,), dtype=input_ids.dtype) * self.tokenizer.pad_token_id
            padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
        elif sequence_length > self.max_length:
            if self.truncation == "left":
                # actually, left truncation may not be reasonable
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
            elif self.truncation == "error":
                raise NotImplementedError(f"{sequence_length=} is larger than {self.max_length=}")
            else:
                raise NotImplementedError(f"Unknown truncation method {self.truncation}")

        position_ids = compute_position_id_with_mask(attention_mask)

        loss_mask = attention_mask.clone()
        if prompt_length > 1:
            # mask out prompt for SFT.
            loss_mask[: min(prompt_length, loss_mask.size(0)) - 1] = 0
        # mask out the last token in response
        loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }


class VisionSFTDataset(SFTDataset):
    """
    Single-turn multimodal SFT dataset for prompt/response/image parquet rows.

    Expected columns:
      - prompt: user prompt string containing one or more ``<image>`` markers
      - response: assistant response string
      - images: list of image objects accepted by ``vision_utils.process_image``
    """

    def __init__(self, parquet_files: Union[str, List[str]], tokenizer, processor: ProcessorMixin, config):
        self.processor = processor
        self.image_key = config.get("image_key", "images")
        super().__init__(parquet_files=parquet_files, tokenizer=tokenizer, config=config)

    @staticmethod
    def _normalize_images(images):
        if images is None:
            return []
        if isinstance(images, np.ndarray):
            return images.tolist()
        if isinstance(images, (list, tuple)):
            return list(images)
        return [images]

    @staticmethod
    def _to_content_segments(prompt: str, has_images: bool):
        if not has_images:
            return prompt
        content = []
        for segment in re.split("(<image>)", str(prompt)):
            if segment == "<image>":
                content.append({"type": "image"})
            elif segment:
                content.append({"type": "text", "text": segment})
        return content

    def __getitem__(self, item):
        from verl.utils.dataset.vision_utils import process_image

        tokenizer = self.tokenizer
        prompt = str(self.prompts[item])
        response = str(self.responses[item])
        row = self.dataframe.iloc[item].to_dict()
        images = self._normalize_images(row.get(self.image_key))
        processed_images = [process_image(image) for image in images]

        placeholder_count = prompt.count("<image>")
        if processed_images and placeholder_count == 0:
            prompt = ("<image>\n" * len(processed_images)) + prompt
            placeholder_count = len(processed_images)
        if placeholder_count != len(processed_images):
            raise ValueError(
                f"Vision SFT sample {item} has {placeholder_count} <image> placeholder(s) "
                f"but {len(processed_images)} image(s)."
            )

        prompt_chat = [{
            "role": "user",
            "content": self._to_content_segments(prompt, bool(processed_images)),
        }]
        prompt_chat_str = self.processor.apply_chat_template(
            prompt_chat,
            add_generation_prompt=True,
            tokenize=False,
            **self.apply_chat_template_kwargs,
        )
        response_chat_str = response + tokenizer.eos_token
        full_text = prompt_chat_str + response_chat_str

        prompt_inputs = self.processor(
            text=[prompt_chat_str],
            images=processed_images or None,
            return_tensors="pt",
        )
        model_inputs = self.processor(
            text=[full_text],
            images=processed_images or None,
            return_tensors="pt",
        )
        prompt_length = int(prompt_inputs["attention_mask"][0].sum().item())
        response_length = int(model_inputs["attention_mask"][0].sum().item()) - prompt_length

        input_ids = model_inputs.pop("input_ids")[0]
        attention_mask = model_inputs.pop("attention_mask")[0]
        model_inputs.pop("second_per_grid_ts", None)

        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            pad_length = self.max_length - sequence_length
            padded_input_ids = torch.full(
                (pad_length,),
                tokenizer.pad_token_id,
                dtype=input_ids.dtype,
            )
            padded_attention_mask = torch.zeros(
                (pad_length,),
                dtype=attention_mask.dtype,
            )
            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
        elif sequence_length > self.max_length:
            if self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
            elif self.truncation == "left":
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
                prompt_length = max(0, prompt_length - (sequence_length - self.max_length))
            elif self.truncation == "error":
                raise NotImplementedError(f"{sequence_length=} is larger than {self.max_length=}")
            else:
                raise NotImplementedError(f"Unknown truncation method {self.truncation}")

        if "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                attention_mask=attention_mask,
            )
            valid_mask = attention_mask.bool()
            text_position_ids = torch.ones((1, len(input_ids)), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)
        else:
            position_ids = compute_position_id_with_mask(attention_mask.unsqueeze(0))[0]

        loss_mask = attention_mask.clone()
        if prompt_length > 1:
            loss_mask[: min(prompt_length, loss_mask.size(0)) - 1] = 0
        if response_length > 0:
            loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0

        sample = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }
        for key, value in dict(model_inputs).items():
            if torch.is_tensor(value):
                sample[key] = value.squeeze(0) if value.dim() > 0 and value.size(0) == 1 else value
        return sample
