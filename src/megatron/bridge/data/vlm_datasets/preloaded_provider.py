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

"""
Provider for datasets preloaded from JSON/JSONL files into conversation schema.
"""

import json
import logging
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

from transformers import AutoProcessor

from megatron.bridge.data.vlm_datasets.conversation_dataset import VLMConversationDataset
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo
from megatron.bridge.training.config import DatasetBuildContext, DatasetProvider


try:
    from qwen_vl_utils import process_vision_info

    HAVE_QWEN_VL_UTILS = True
except ImportError:
    HAVE_QWEN_VL_UTILS = False

ToolCallFormat = Literal["auto", "qwen_agent", "hermes", "none"]

_LOGGER = logging.getLogger(__name__)
_TOOL_CALL_ROLES = {"tool_call", "function_call"}
_TOOL_RESPONSE_ROLES = {"tool", "tool_response", "function"}
_QWEN_MIN_PIXELS = 200704
_QWEN_MAX_PIXELS = 1003520
_OVERLENGTH_LOG_LIMIT = 8


def _split_text_by_placeholders(
    text: str, image_paths: List[str], video_paths: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Split legacy text containing "<image>"/"<video>" markers into an alternating
    sequence of text and media parts, preserving the original order and spacing.
    """
    parts: List[Dict[str, Any]] = []
    img_idx = 0
    vid_idx = 0

    last_end = 0
    for match in re.finditer(r"<image>|<video>", text):
        # Preceding text (if any)
        if match.start() > last_end:
            seg = text[last_end : match.start()]
            if seg:
                parts.append({"type": "text", "text": seg})

        token = match.group(0)
        if token == "<image>":
            if img_idx >= len(image_paths):
                _LOGGER.warning("Encountered <image> without corresponding entry in images list.")
            else:
                parts.append({"type": "image", "image": image_paths[img_idx]})
            img_idx += 1
        else:  # <video>
            if video_paths is None or vid_idx >= len(video_paths):
                _LOGGER.warning("Encountered <video> without corresponding entry in videos list.")
            else:
                parts.append({"type": "video", "video": video_paths[vid_idx]})
            vid_idx += 1
        last_end = match.end()

    # Trailing text (if any)
    if last_end < len(text):
        tail = text[last_end:]
        if tail:
            parts.append({"type": "text", "text": tail})
    return parts


def _normalize_media_path(path: Any, base_folder: Optional[str]) -> Any:
    if not isinstance(path, str) or base_folder is None:
        return path
    if path.startswith(("http:", "https:", "file:")) or os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_folder, path))


def _normalize_paths(paths: Optional[List[Any]], base_folder: Optional[str]) -> Optional[List[Any]]:
    if not paths or base_folder is None:
        return paths
    return [_normalize_media_path(p, base_folder) for p in paths]


def _json_loads_if_string(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def _normalize_role(role: Optional[str]) -> Optional[str]:
    if role is None:
        return None
    return role.lower().replace("-", "_")


def _message_role_and_content(message: Dict[str, Any]) -> Tuple[str, Any]:
    role = _normalize_role(message.get("role"))
    if role is None:
        from_role = str(message.get("from", "human")).lower()
        role = "user" if from_role in ("human", "user") else "assistant"
        return role, message.get("value", "")
    return role, message.get("content", "")


def _record_has_tool_calling(record: Dict[str, Any], source_msgs: List[Dict[str, Any]]) -> bool:
    if record.get("tools") is not None:
        return True
    for msg in source_msgs:
        role, _ = _message_role_and_content(msg)
        if role in _TOOL_CALL_ROLES or role in _TOOL_RESPONSE_ROLES or msg.get("tool_calls"):
            return True
    return False


def _normalize_tool(tool: Any) -> Dict[str, Any]:
    tool_obj = _json_loads_if_string(tool)
    if not isinstance(tool_obj, dict):
        raise ValueError("Tool definitions must be JSON objects.")
    if "type" not in tool_obj and "function" not in tool_obj:
        return {"type": "function", "function": tool_obj}
    return tool_obj


def _normalize_tools(tools: Any) -> List[Dict[str, Any]]:
    if tools is None:
        return []
    tools_obj = _json_loads_if_string(tools)
    if isinstance(tools_obj, dict):
        tool_items = [tools_obj]
    elif isinstance(tools_obj, list):
        tool_items = tools_obj
    else:
        raise ValueError("The tools field must be a JSON object, JSON array, or JSON string.")
    return [_normalize_tool(tool) for tool in tool_items]


def _format_hermes_tools(tools: List[Dict[str, Any]], system: str) -> str:
    tool_descs = "\n".join(json.dumps(tool, ensure_ascii=False) for tool in tools)
    return (
        f"{system}\n\n"
        "# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        "<tools>\n"
        f"{tool_descs}\n"
        "</tools>\n\n"
        "For each function call, return a json object with function name and arguments within "
        "<tool_call></tool_call> XML tags:\n"
        "<tool_call>\n"
        '{"name": <function-name>, "arguments": <args-json-object>}\n'
        "</tool_call>"
    )


def _normalize_tool_call(tool_call: Any) -> Dict[str, Any]:
    tool_call_obj = _json_loads_if_string(tool_call)
    if not isinstance(tool_call_obj, dict):
        raise ValueError("Tool call content must be a JSON object or JSON string.")

    function = tool_call_obj.get("function")
    if isinstance(function, dict):
        name = function.get("name") or tool_call_obj.get("name")
        arguments = _json_loads_if_string(function.get("arguments", {}))
    else:
        name = tool_call_obj.get("name")
        arguments = _json_loads_if_string(tool_call_obj.get("arguments", {}))

    if not name:
        raise ValueError("Tool call content must include a function name.")
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise ValueError("Tool call arguments must be a JSON object.")
    return {"name": name, "arguments": arguments}


def _normalize_tool_call_arguments(arguments: Any) -> Dict[str, Any]:
    arguments = _json_loads_if_string(arguments)
    if arguments is None:
        return {}
    if not isinstance(arguments, dict):
        raise ValueError("Tool call arguments must be a JSON object.")
    return arguments


def _normalize_qwen_agent_tool_call(tool_call: Any, *, fallback_id: str) -> Dict[str, Any]:
    tool_call_obj = _json_loads_if_string(tool_call)
    if not isinstance(tool_call_obj, dict):
        raise ValueError("Tool call content must be a JSON object or JSON string.")

    function = tool_call_obj.get("function")
    if isinstance(function, dict):
        name = function.get("name") or tool_call_obj.get("name")
        arguments = _json_loads_if_string(function.get("arguments", {}))
    else:
        name = tool_call_obj.get("name")
        arguments = _json_loads_if_string(tool_call_obj.get("arguments", {}))

    if not name:
        raise ValueError("Tool call content must include a function name.")
    return {
        "id": str(tool_call_obj.get("id") or tool_call_obj.get("tool_call_id") or fallback_id),
        "type": "function",
        "function": {
            "name": name,
            # Qwen3.5's HF chat_template iterates function.arguments|items to render
            # <parameter=...> blocks, so keep arguments as a mapping instead of the
            # OpenAI wire-format JSON string.
            "arguments": _normalize_tool_call_arguments(arguments),
        },
    }


def _iter_tool_calls(message: Dict[str, Any]) -> List[Any]:
    tool_calls = message.get("tool_calls")
    if tool_calls is None:
        return [message.get("content")]
    if isinstance(tool_calls, list):
        return tool_calls
    return [tool_calls]


def _format_hermes_tool_calls(messages: List[Dict[str, Any]]) -> str:
    blocks: List[str] = []
    for message in messages:
        for tool_call in _iter_tool_calls(message):
            normalized = _normalize_tool_call(tool_call)
            blocks.append("<tool_call>\n" + json.dumps(normalized, ensure_ascii=False) + "\n</tool_call>")
    return "\n".join(blocks)


def _format_hermes_tool_responses(messages: List[Dict[str, Any]]) -> str:
    blocks = []
    for message in messages:
        blocks.append("<tool_response>\n" + _content_to_text(message.get("content", "")) + "\n</tool_response>")
    return "\n".join(blocks)


def _media_path_from_content_part(part: Dict[str, Any], media_type: str) -> Any:
    value = part.get(media_type) or part.get("path")
    if value is not None:
        return value
    if media_type == "image":
        image_url = part.get("image_url")
        if isinstance(image_url, dict):
            return image_url.get("url") or image_url.get("path")
        return image_url
    return None


def _split_text_by_placeholders_with_cursors(
    text: str,
    image_paths: List[Any],
    video_paths: List[Any],
    image_cursor: List[int],
    video_cursor: List[int],
    *,
    strict: bool,
) -> List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = []
    last_end = 0

    for match in re.finditer(r"<image>|<video>", text):
        if match.start() > last_end:
            seg = text[last_end : match.start()]
            if seg:
                parts.append({"type": "text", "text": seg})

        token = match.group(0)
        if token == "<image>":
            if image_cursor[0] >= len(image_paths):
                message = "Encountered <image> without corresponding entry in images list."
                if strict:
                    raise ValueError(message)
                _LOGGER.warning(message)
            else:
                parts.append({"type": "image", "image": image_paths[image_cursor[0]]})
            image_cursor[0] += 1
        else:
            if video_cursor[0] >= len(video_paths):
                message = "Encountered <video> without corresponding entry in videos list."
                if strict:
                    raise ValueError(message)
                _LOGGER.warning(message)
            else:
                parts.append({"type": "video", "video": video_paths[video_cursor[0]]})
            video_cursor[0] += 1
        last_end = match.end()

    if last_end < len(text):
        tail = text[last_end:]
        if tail:
            parts.append({"type": "text", "text": tail})
    return parts


def _extend_content_parts(
    target: List[Dict[str, Any]],
    parts: List[Dict[str, Any]],
    *,
    text_separator: str = "",
) -> None:
    separator = text_separator
    for part in parts:
        if target and target[-1].get("type") == "text" and part.get("type") == "text":
            existing = target[-1].get("text", "")
            incoming = part.get("text", "")
            if existing and incoming and separator:
                target[-1]["text"] = existing + separator + incoming
            else:
                target[-1]["text"] = existing + incoming
        else:
            target.append(part)
        separator = ""


def _consume_next_media(
    media_paths: List[Any],
    cursor: List[int],
    *,
    media_type: str,
    strict: bool,
) -> Optional[Any]:
    if cursor[0] >= len(media_paths):
        message = f"Encountered {media_type} content part without corresponding top-level {media_type}s entry."
        if strict:
            raise ValueError(message)
        _LOGGER.warning(message)
        return None
    media_path = media_paths[cursor[0]]
    cursor[0] += 1
    return media_path


def _content_to_parts(
    content: Any,
    image_paths: List[Any],
    video_paths: List[Any],
    image_cursor: List[int],
    video_cursor: List[int],
    *,
    image_folder: Optional[str],
    strict: bool,
) -> List[Dict[str, Any]]:
    if isinstance(content, str):
        return _split_text_by_placeholders_with_cursors(
            content, image_paths, video_paths, image_cursor, video_cursor, strict=strict
        )
    if isinstance(content, dict):
        content_items = [content]
    elif isinstance(content, list):
        content_items = content
    elif content is None:
        content_items = []
    else:
        content_items = [str(content)]

    parts: List[Dict[str, Any]] = []
    for item in content_items:
        if isinstance(item, str):
            item_parts = _split_text_by_placeholders_with_cursors(
                item, image_paths, video_paths, image_cursor, video_cursor, strict=strict
            )
            _extend_content_parts(parts, item_parts)
            continue
        if not isinstance(item, dict):
            _extend_content_parts(parts, [{"type": "text", "text": _content_to_text(item)}])
            continue

        item_type = item.get("type")
        if item_type == "text" or (item_type is None and "text" in item):
            item_parts = _split_text_by_placeholders_with_cursors(
                item.get("text", ""), image_paths, video_paths, image_cursor, video_cursor, strict=strict
            )
            _extend_content_parts(parts, item_parts)
        elif item_type in ("image", "image_url") or "image" in item or "image_url" in item:
            image_path = _media_path_from_content_part(item, "image")
            if image_path is None:
                image_path = _consume_next_media(image_paths, image_cursor, media_type="image", strict=strict)
            if image_path is not None:
                parts.append({"type": "image", "image": _normalize_media_path(image_path, image_folder)})
        elif item_type == "video" or "video" in item:
            video_path = _media_path_from_content_part(item, "video")
            if video_path is None:
                video_path = _consume_next_media(video_paths, video_cursor, media_type="video", strict=strict)
            if video_path is not None:
                parts.append({"type": "video", "video": _normalize_media_path(video_path, image_folder)})
        else:
            _extend_content_parts(parts, [{"type": "text", "text": _content_to_text(item)}])
    return parts


def _append_conversation_message(
    conversation: List[Dict[str, Any]],
    role: str,
    content_parts: List[Dict[str, Any]],
) -> None:
    if not content_parts:
        content_parts = [{"type": "text", "text": ""}]
    if conversation and conversation[-1]["role"] == role and role in ("user", "assistant"):
        _extend_content_parts(conversation[-1]["content"], content_parts, text_separator="\n")
    else:
        conversation.append({"role": role, "content": content_parts})


def _record_media(record: Dict[str, Any], image_folder: Optional[str]) -> Tuple[List[Any], List[Any]]:
    images: List[Any] = []
    if "images" in record and isinstance(record["images"], list):
        images = record["images"]
    elif "image" in record and record["image"] is not None:
        if isinstance(record["image"], list):
            images = record["image"]
        else:
            images = [record["image"]]
    videos: List[Any] = record.get("videos", []) or []
    return _normalize_paths(images, image_folder) or [], _normalize_paths(videos, image_folder) or []


def _validate_media_cursors(
    image_cursor: List[int],
    images: List[Any],
    video_cursor: List[int],
    videos: List[Any],
) -> None:
    if image_cursor[0] != len(images):
        raise ValueError(
            f"Image placeholder count mismatch: consumed {image_cursor[0]} entries but images has {len(images)}."
        )
    if video_cursor[0] != len(videos):
        raise ValueError(
            f"Video placeholder count mismatch: consumed {video_cursor[0]} entries but videos has {len(videos)}."
        )


def _record_to_tool_aware_conversation(
    record: Dict[str, Any],
    image_folder: Optional[str],
    source_msgs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Render Swift/Hermes-style tool-call records into Qwen chat-template conversations."""
    tools = _normalize_tools(record.get("tools"))
    images, videos = _record_media(record, image_folder)
    image_cursor = [0]
    video_cursor = [0]

    conversation: List[Dict[str, Any]] = []
    start_idx = 0
    system_content = ""
    if source_msgs:
        first_role, first_content = _message_role_and_content(source_msgs[0])
        if first_role == "system":
            system_content = _content_to_text(first_content)
            start_idx = 1

    if tools:
        conversation.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": _format_hermes_tools(tools, system_content)}],
            }
        )
    elif system_content:
        conversation.append({"role": "system", "content": [{"type": "text", "text": system_content}]})

    pending_tool_calls: List[Dict[str, Any]] = []
    pending_tool_responses: List[Dict[str, Any]] = []

    def flush_tool_calls() -> None:
        if not pending_tool_calls:
            return
        _append_conversation_message(
            conversation,
            "assistant",
            [{"type": "text", "text": _format_hermes_tool_calls(pending_tool_calls)}],
        )
        pending_tool_calls.clear()

    def flush_tool_responses() -> None:
        if not pending_tool_responses:
            return
        content = _format_hermes_tool_responses(pending_tool_responses)
        parts = _content_to_parts(
            content, images, videos, image_cursor, video_cursor, image_folder=image_folder, strict=True
        )
        _append_conversation_message(conversation, "user", parts)
        pending_tool_responses.clear()

    for msg in source_msgs[start_idx:]:
        role, content = _message_role_and_content(msg)
        if msg.get("tool_calls"):
            flush_tool_calls()
            flush_tool_responses()
            if role == "assistant" and content:
                parts = _content_to_parts(
                    content, images, videos, image_cursor, video_cursor, image_folder=image_folder, strict=True
                )
                _append_conversation_message(conversation, "assistant", parts)
            pending_tool_calls.append(msg)
            continue

        if role in _TOOL_CALL_ROLES:
            flush_tool_responses()
            pending_tool_calls.append(msg)
            continue

        if role in _TOOL_RESPONSE_ROLES:
            flush_tool_calls()
            pending_tool_responses.append(msg)
            continue

        flush_tool_calls()
        flush_tool_responses()

        if role not in ("system", "user", "assistant"):
            raise ValueError(f"Unsupported role in tool-aware preloaded record: {role}")
        parts = _content_to_parts(
            content, images, videos, image_cursor, video_cursor, image_folder=image_folder, strict=True
        )
        _append_conversation_message(conversation, role, parts)

    flush_tool_calls()
    flush_tool_responses()
    _validate_media_cursors(image_cursor, images, video_cursor, videos)
    return conversation


def _append_qwen_agent_tool_calls(
    conversation: List[Dict[str, Any]],
    tool_calls: List[Dict[str, Any]],
) -> None:
    if conversation and conversation[-1]["role"] == "assistant":
        conversation[-1].setdefault("tool_calls", []).extend(tool_calls)
    else:
        conversation.append({"role": "assistant", "content": [], "tool_calls": tool_calls})


def _record_to_qwen_agent_conversation(
    record: Dict[str, Any],
    image_folder: Optional[str],
    source_msgs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Render Qwen-Agent records into Qwen3.5 chat-template-compatible tool calls."""
    images, videos = _record_media(record, image_folder)
    image_cursor = [0]
    video_cursor = [0]

    conversation: List[Dict[str, Any]] = []
    pending_tool_calls: List[Dict[str, Any]] = []
    pending_tool_call_ids: List[str] = []

    def flush_tool_calls() -> None:
        if not pending_tool_calls:
            return
        qwen_agent_tool_calls: List[Dict[str, Any]] = []
        for message in pending_tool_calls:
            for raw_tool_call in _iter_tool_calls(message):
                fallback_id = str(len(qwen_agent_tool_calls) + 1)
                tool_call = _normalize_qwen_agent_tool_call(raw_tool_call, fallback_id=fallback_id)
                qwen_agent_tool_calls.append(tool_call)
                pending_tool_call_ids.append(tool_call["id"])
        _append_qwen_agent_tool_calls(conversation, qwen_agent_tool_calls)
        pending_tool_calls.clear()

    def next_tool_call_id(message: Dict[str, Any]) -> str:
        if message.get("id") or message.get("tool_call_id"):
            return str(message.get("id") or message.get("tool_call_id"))
        if pending_tool_call_ids:
            return pending_tool_call_ids.pop(0)
        return "1"

    def append_tool_response(message: Dict[str, Any]) -> None:
        content = message.get("content", "")
        parts = _content_to_parts(
            content, images, videos, image_cursor, video_cursor, image_folder=image_folder, strict=True
        )
        tool_message = {
            "role": "tool",
            "content": parts,
            "id": next_tool_call_id(message),
        }
        if message.get("name"):
            tool_message["name"] = message["name"]
        conversation.append(tool_message)

    for msg in source_msgs:
        role, content = _message_role_and_content(msg)
        if msg.get("tool_calls"):
            flush_tool_calls()
            if role == "assistant" and content:
                parts = _content_to_parts(
                    content, images, videos, image_cursor, video_cursor, image_folder=image_folder, strict=True
                )
                _append_conversation_message(conversation, "assistant", parts)
            pending_tool_calls.append(msg)
            continue

        if role in _TOOL_CALL_ROLES:
            pending_tool_calls.append(msg)
            continue

        if role in _TOOL_RESPONSE_ROLES:
            flush_tool_calls()
            append_tool_response(msg)
            continue

        flush_tool_calls()

        if role not in ("system", "user", "assistant"):
            raise ValueError(f"Unsupported role in Qwen-Agent preloaded record: {role}")
        parts = _content_to_parts(
            content, images, videos, image_cursor, video_cursor, image_folder=image_folder, strict=True
        )
        _append_conversation_message(conversation, role, parts)

    flush_tool_calls()
    _validate_media_cursors(image_cursor, images, video_cursor, videos)
    return conversation


def _record_to_conversation(
    record: Dict[str, Any],
    image_folder: Optional[str],
    *,
    tool_call_format: ToolCallFormat = "auto",
) -> Optional[List[Dict[str, Any]]]:
    """
    Transform a single legacy record into an AutoProcessor-friendly conversation schema.
    Supports two input styles:
      - {"conversation": [...]} already in HF schema -> passthrough
      - {"messages": [...], "images": [...], "videos": [...]} with <image>/<video> markers
      - Qwen3.5/Swift-style {"tools": [...], "messages": [...]} with tool_call/tool roles
    """
    if tool_call_format not in ("auto", "qwen_agent", "hermes", "none"):
        raise ValueError(f"Unsupported tool_call_format: {tool_call_format}")
    if "conversation" in record:
        return record["conversation"]

    # Accept legacy "messages" or LLaVA-style "conversations"
    messages = record.get("messages")
    llava_conversations = record.get("conversations")
    if not messages and not llava_conversations:
        return None

    source_msgs = messages if messages is not None else llava_conversations
    if source_msgs is None:
        return None
    if tool_call_format == "hermes":
        return _record_to_tool_aware_conversation(record, image_folder, source_msgs)
    if tool_call_format == "qwen_agent" or (
        tool_call_format == "auto" and _record_has_tool_calling(record, source_msgs)
    ):
        return _record_to_qwen_agent_conversation(record, image_folder, source_msgs)

    # Build images/videos list from several possible fields
    images, videos = _record_media(record, image_folder)

    conversation: List[Dict[str, Any]] = []
    for msg in source_msgs:
        # LLaVA uses {'from': 'human'|'gpt', 'value': '...'}
        role, content_str = _message_role_and_content(msg)

        content_list = _split_text_by_placeholders(content_str, images, videos)
        if content_list:
            # Reorder to media-first followed by a single combined text segment to
            # match typical VLM chat templates (media before text)
            media_parts = [p for p in content_list if p.get("type") in ("image", "video")]
            text_parts = [p.get("text", "") for p in content_list if p.get("type") == "text" and p.get("text")]
            if text_parts:
                media_parts.append({"type": "text", "text": "".join(text_parts)})
            content_list = media_parts
        if not content_list:
            content_list = [{"type": "text", "text": content_str}]
        conversation.append({"role": role, "content": content_list})
    return conversation


def _load_preloaded_examples(path: str) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    if path.endswith(".jsonl"):
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                examples.append(json.loads(line))
    else:
        with open(path, "r") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            examples = payload
        elif isinstance(payload, dict):
            # Some datasets wrap under a key, try common ones
            for key in ["data", "examples", "records"]:
                if key in payload and isinstance(payload[key], list):
                    examples = payload[key]
                    break
            if not examples:
                examples = [payload]
        else:
            raise ValueError(f"Unsupported JSON structure in {path}")
    return examples


def _record_to_preloaded_example(
    record: Dict[str, Any],
    image_folder: Optional[str],
    *,
    tool_call_format: ToolCallFormat,
) -> Optional[Dict[str, Any]]:
    conv = _record_to_conversation(record, image_folder, tool_call_format=tool_call_format)
    if conv is None:
        return None

    base_example: Dict[str, Any] = {"conversation": conv}
    messages = record.get("messages")
    llava_conversations = record.get("conversations")
    source_msgs = messages if messages is not None else llava_conversations
    if source_msgs is not None and tool_call_format in ("auto", "qwen_agent"):
        if _record_has_tool_calling(record, source_msgs):
            tools = _normalize_tools(record.get("tools"))
            if tools:
                base_example["tools"] = tools
    return base_example


def _apply_chat_template_for_preloaded_example(processor: Any, example: Dict[str, Any], **kwargs: Any) -> str:
    tools = example.get("tools")
    if tools:
        kwargs["tools"] = tools
    return processor.apply_chat_template(example["conversation"], **kwargs)


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _preloaded_example_has_media(example: Dict[str, Any]) -> bool:
    for message in example.get("conversation", []):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") in ("image", "video"):
                return True
    return False


def _tokenized_length_for_preloaded_example(processor: Any, example: Dict[str, Any]) -> int:
    text = _apply_chat_template_for_preloaded_example(processor, example, tokenize=False)
    processor_kwargs: Dict[str, Any] = {
        "text": [text],
        "padding": False,
        "return_tensors": "pt",
    }

    if _preloaded_example_has_media(example):
        if not HAVE_QWEN_VL_UTILS:
            raise ImportError("qwen_vl_utils is required to length-filter preloaded examples with images/videos.")
        images, videos = process_vision_info(example["conversation"])
        images = _as_list(images)
        videos = _as_list(videos)
        if images:
            processor_kwargs["images"] = [images]
            processor_kwargs["min_pixels"] = _QWEN_MIN_PIXELS
            processor_kwargs["max_pixels"] = _QWEN_MAX_PIXELS
        if videos:
            processor_kwargs["videos"] = [videos]

    batch = processor(**processor_kwargs)
    input_ids = batch["input_ids"]
    if hasattr(input_ids, "shape"):
        return int(input_ids.shape[-1])
    if input_ids and isinstance(input_ids[0], list):
        return len(input_ids[0])
    return len(input_ids)


def _split_base_examples_for_validation(
    base_examples: List[Dict[str, Any]],
    validation_split_ratio: float,
    validation_split_seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if validation_split_ratio <= 0.0:
        return base_examples, []
    if validation_split_ratio >= 1.0:
        raise ValueError("validation_split_ratio must be less than 1.0")
    if len(base_examples) < 2:
        _LOGGER.warning(
            "Preloaded VLM validation split requested, but only %s usable example(s) remain; "
            "keeping all examples in train.",
            len(base_examples),
        )
        return base_examples, []

    validation_count = max(1, int(len(base_examples) * validation_split_ratio + 0.5))
    validation_count = min(validation_count, len(base_examples) - 1)

    indices = list(range(len(base_examples)))
    random.Random(validation_split_seed).shuffle(indices)
    validation_indices = set(indices[:validation_count])

    train_examples = [example for idx, example in enumerate(base_examples) if idx not in validation_indices]
    validation_examples = [example for idx, example in enumerate(base_examples) if idx in validation_indices]
    return train_examples, validation_examples


@dataclass(kw_only=True)
class PreloadedVLMConversationProvider(DatasetProvider):
    """DatasetProvider that builds VLM conversation datasets from preloaded JSON/JSONL files.

    The provider converts legacy Qwen2/VL style records with '<image>'/'<video>' markers
    into a conversation schema consumable by HuggingFace AutoProcessor for Qwen2.5-VL.
    It can also render Swift-style Qwen3.5 tool_call/tool roles into the
    Qwen-Agent raw API shape with sample-level tools, assistant tool_calls, and
    tool response messages.
    """

    # Required to match model.seq_length
    seq_length: int

    # HF processor/model identifier (e.g., "Qwen/Qwen2.5-VL-3B-Instruct")
    hf_processor_path: str = "Qwen/Qwen2.5-VL-3B-Instruct"

    # Paths to preloaded datasets (JSON/JSONL). Any can be None.
    train_data_path: Optional[str] = None
    valid_data_path: Optional[str] = None
    test_data_path: Optional[str] = None

    # Optional image/video root to resolve relative paths
    image_folder: Optional[str] = None

    # Render tool_call/tool roles. "auto" follows Qwen3.5 chat-template format for tool-aware records.
    tool_call_format: ToolCallFormat = "auto"

    # Keep parity with GPTDatasetConfig usage in batching utilities
    skip_getting_attention_mask_from_dataset: bool = True

    # Default dataloader type for VLM providers
    dataloader_type: Optional[Literal["single", "cyclic", "external"]] = "single"

    # Enable batch-level online sequence packing
    pack_sequences_in_batch: bool = False

    # Drop examples whose processor-rendered tokenized length exceeds seq_length.
    drop_overlength_samples: bool = False

    # Split this ratio from train_data_path into validation when valid_data_path is unset.
    validation_split_ratio: float = 0.0
    validation_split_seed: int = 1234

    def _build_base_examples(
        self,
        raw_examples: List[Dict[str, Any]],
        split_name: str,
        processor: Any,
    ) -> List[Dict[str, Any]]:
        base_examples: List[Dict[str, Any]] = []
        dropped_overlength = 0
        overlength_samples: List[Tuple[int, int]] = []
        max_overlength = 0
        for raw_idx, rec in enumerate(raw_examples, start=1):
            base_example = _record_to_preloaded_example(rec, self.image_folder, tool_call_format=self.tool_call_format)
            if base_example is None:
                continue
            if self.drop_overlength_samples:
                tokenized_length = _tokenized_length_for_preloaded_example(processor, base_example)
                if tokenized_length > self.seq_length:
                    dropped_overlength += 1
                    max_overlength = max(max_overlength, tokenized_length)
                    if len(overlength_samples) < _OVERLENGTH_LOG_LIMIT:
                        overlength_samples.append((raw_idx, tokenized_length))
                    continue
            base_examples.append(base_example)
        if self.drop_overlength_samples:
            _LOGGER.info(
                "Preloaded VLM overlength filter for %s: kept=%s dropped=%s seq_length=%s "
                "max_dropped_length=%s sample_drops=%s",
                split_name,
                len(base_examples),
                dropped_overlength,
                self.seq_length,
                max_overlength if dropped_overlength else None,
                overlength_samples,
            )
        return base_examples

    def _build_dataset_from_base_examples(
        self,
        base_examples: List[Dict[str, Any]],
        split_name: str,
        target_length: int,
        processor: Any,
    ) -> Optional[VLMConversationDataset]:
        if target_length <= 0:
            return None
        if not base_examples:
            _LOGGER.warning(f"No usable examples parsed from {split_name}")
            return None
        # Pass seq_length as max_length so the collate function hard-truncates
        # sequences that exceed the model context window.  This is the framework-
        # side fix for Pitfall #23: without max_length the processor falls back to
        # tokenizer.model_max_length (~128K) and overlength multimodal samples
        # (vision_tokens + text_tokens > seq_length) reach PP stage-0 unchanged,
        # causing tensor shape mismatches or attention OOM.
        return VLMConversationDataset(
            base_examples=base_examples,
            target_length=target_length,
            processor=processor,
            max_length=self.seq_length,
        )

    def _build_split_dataset(
        self,
        split_path: Optional[str],
        target_length: int,
        processor: Any,
    ) -> Optional[VLMConversationDataset]:
        if not split_path or target_length <= 0:
            return None
        raw_examples = _load_preloaded_examples(split_path)
        base_examples = self._build_base_examples(raw_examples, split_path, processor)
        return self._build_dataset_from_base_examples(base_examples, split_path, target_length, processor)

    def build_datasets(self, context: DatasetBuildContext) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
        processor = AutoProcessor.from_pretrained(
            self.hf_processor_path,
            trust_remote_code=is_safe_repo(
                trust_remote_code=self.trust_remote_code,
                hf_path=self.hf_processor_path,
            ),
        )
        if self.valid_data_path is None and self.train_data_path is not None and self.validation_split_ratio > 0.0:
            base_examples = self._build_base_examples(
                _load_preloaded_examples(self.train_data_path), self.train_data_path, processor
            )
            train_examples, validation_examples = _split_base_examples_for_validation(
                base_examples, self.validation_split_ratio, self.validation_split_seed
            )
            _LOGGER.info(
                "Preloaded VLM validation split from %s: train=%s validation=%s ratio=%s seed=%s",
                self.train_data_path,
                len(train_examples),
                len(validation_examples),
                self.validation_split_ratio,
                self.validation_split_seed,
            )
            train_ds = self._build_dataset_from_base_examples(
                train_examples, f"{self.train_data_path}:train", context.train_samples, processor
            )
            valid_ds = self._build_dataset_from_base_examples(
                validation_examples, f"{self.train_data_path}:validation", context.valid_samples, processor
            )
        else:
            train_ds = self._build_split_dataset(self.train_data_path, context.train_samples, processor)
            valid_ds = self._build_split_dataset(self.valid_data_path, context.valid_samples, processor)
        test_ds = self._build_split_dataset(self.test_data_path, context.test_samples, processor)
        return train_ds, valid_ds, test_ds
