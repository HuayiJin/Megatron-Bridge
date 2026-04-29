# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import json
import os
import tempfile

import pytest
import torch

import megatron.bridge.data.vlm_datasets.preloaded_provider as pre
from megatron.bridge.training.config import DatasetBuildContext


class _DummyProcessor:
    class _Tok:
        pad_token_id = 0
        eos_token_id = 2
        added_tokens_decoder = {}

        def __call__(self, text, add_special_tokens=False):  # noqa: ARG002 - parity with HF
            return {"input_ids": [1, 2, 3]}

    def __init__(self):
        self.tokenizer = self._Tok()

    def apply_chat_template(self, conversation, tokenize=False, **kwargs):  # noqa: ARG002
        if tokenize:
            return {"input_ids": torch.tensor([[1, 2, 3]]), "pixel_values": torch.randn(1, 1, 3, 4, 4)}
        return "dummy"

    def __call__(self, text=None, images=None, padding=True, return_tensors="pt", **kwargs):  # noqa: ARG002
        out = {"input_ids": torch.tensor([[1, 2, 3]])}
        if images is not None:
            n = len(images)
            out["pixel_values"] = torch.randn(1, n, 3, 4, 4)
            out["image_grid_thw"] = torch.tensor([[[1, 2, 2]] * n])
        return out


def _write_tmp_jsonl(rows):
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


def test_split_text_by_placeholders_basic():
    parts = pre._split_text_by_placeholders("a<image>b<video>c", ["img.png"], ["vid.mp4"])  # noqa: SLF001
    types = [p["type"] for p in parts]
    assert types == ["text", "image", "text", "video", "text"]


def test_record_to_conversation_legacy_and_llava(tmp_path):  # noqa: ARG001 - tmp_path reserved
    conv = pre._record_to_conversation(  # noqa: SLF001
        {
            "messages": [
                {"role": "user", "content": "hello <image>"},
                {"role": "assistant", "content": "world"},
            ],
            "image": "rel/img.png",
        },
        image_folder="/abs",
    )
    assert isinstance(conv, list) and conv[0]["content"][1]["type"] == "text"

    # LLaVA-style
    conv2 = pre._record_to_conversation(  # noqa: SLF001
        {
            "conversations": [
                {"from": "human", "value": "<image> say x"},
                {"from": "gpt", "value": "x"},
            ],
            "images": ["a.png"],
        },
        image_folder=None,
    )
    assert conv2[0]["role"] == "user"


def test_record_to_conversation_qwen_agent_tool_roles():
    conv = pre._record_to_conversation(  # noqa: SLF001
        {
            "tools": [
                {
                    "name": "search",
                    "description": "Search documents.",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                }
            ],
            "messages": [
                {"role": "system", "content": "You are careful."},
                {"role": "user", "content": "Find papers."},
                {"role": "tool_call", "content": '{"name": "search", "arguments": {"query": "vlm"}}'},
                {"role": "tool_call", "content": '{"name": "search", "arguments": {"query": "tool"}}'},
                {"role": "tool", "content": "first result"},
                {"role": "tool_response", "content": {"ok": True}},
                {"role": "assistant", "content": "Done."},
            ],
        },
        image_folder=None,
    )

    assert [message["role"] for message in conv] == ["system", "user", "assistant", "tool", "tool", "assistant"]
    assert conv[2]["tool_calls"] == [
        {
            "id": "1",
            "type": "function",
            "function": {"name": "search", "arguments": {"query": "vlm"}},
        },
        {
            "id": "2",
            "type": "function",
            "function": {"name": "search", "arguments": {"query": "tool"}},
        },
    ]
    assert conv[3]["id"] == "1"
    assert conv[4]["id"] == "2"
    assert conv[3]["content"][0]["text"] == "first result"
    assert conv[4]["content"][0]["text"] == '{"ok": true}'


def test_record_to_preloaded_example_preserves_qwen_agent_tools():
    example = pre._record_to_preloaded_example(  # noqa: SLF001
        {
            "tools": [{"name": "search", "parameters": {"type": "object", "properties": {}}}],
            "messages": [{"role": "user", "content": "Find papers."}, {"role": "assistant", "content": "Done."}],
        },
        image_folder=None,
        tool_call_format="auto",
    )

    assert example["tools"] == [
        {"type": "function", "function": {"name": "search", "parameters": {"type": "object", "properties": {}}}}
    ]


def test_record_to_conversation_qwen_agent_assigns_media_globally():
    conv = pre._record_to_conversation(  # noqa: SLF001
        {
            "tools": [{"name": "inspect", "parameters": {"type": "object", "properties": {}}}],
            "images": ["first.png", "second.png"],
            "messages": [
                {"role": "user", "content": "look <image>"},
                {"role": "tool_call", "content": '{"name": "inspect", "arguments": {}}'},
                {"role": "tool", "content": "tool saw <image>"},
                {"role": "assistant", "content": "ok"},
            ],
        },
        image_folder="/abs",
    )

    assert conv[1]["content"][1] == {"type": "image", "image": "/abs/first.png"}
    assert {"type": "image", "image": "/abs/second.png"} in conv[3]["content"]
    assert conv[3]["role"] == "tool"


def test_record_to_conversation_qwen_agent_rejects_media_mismatch():
    with pytest.raises(ValueError, match="Image placeholder count mismatch"):
        pre._record_to_conversation(  # noqa: SLF001
            {
                "tools": [{"name": "search", "parameters": {"type": "object", "properties": {}}}],
                "images": ["unused.png"],
                "messages": [{"role": "user", "content": "no media"}, {"role": "assistant", "content": "ok"}],
            },
            image_folder=None,
        )


def test_record_to_conversation_qwen_agent_openai_tool_calls():
    conv = pre._record_to_conversation(  # noqa: SLF001
        {
            "messages": [
                {"role": "user", "content": "find it"},
                {
                    "role": "assistant",
                    "content": "Checking.",
                    "tool_calls": [
                        {"function": {"name": "search", "arguments": '{"query": "qwen"}'}},
                    ],
                },
                {"role": "tool", "content": "hit"},
                {"role": "assistant", "content": "final"},
            ]
        },
        image_folder=None,
    )

    assert [message["role"] for message in conv] == ["user", "assistant", "tool", "assistant"]
    assistant_text = conv[1]["content"][0]["text"]
    assert assistant_text == "Checking."
    assert conv[1]["tool_calls"] == [
        {
            "id": "1",
            "type": "function",
            "function": {"name": "search", "arguments": {"query": "qwen"}},
        }
    ]
    assert conv[2]["id"] == "1"


def test_load_and_build_provider(monkeypatch):
    # Create small jsonl
    rows = [{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]}]
    path = _write_tmp_jsonl(rows)

    # Stub AutoProcessor
    import transformers

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", staticmethod(lambda *a, **k: _DummyProcessor()))

    provider = pre.PreloadedVLMConversationProvider(
        seq_length=16, hf_processor_path="dummy/model", train_data_path=path
    )

    ctx = DatasetBuildContext(train_samples=2, valid_samples=0, test_samples=0)
    train_ds, valid_ds, test_ds = provider.build_datasets(ctx)
    assert train_ds is not None and len(train_ds) == 2
    assert valid_ds is None and test_ds is None
