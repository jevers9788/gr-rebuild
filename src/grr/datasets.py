import itertools
import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer

# ------------------------
# Common utilities
# ------------------------

MENTION_RE = re.compile(r"<@[\w\d_]+>")
URL_RE = re.compile(r"<https?://[^>|]+(?:\|[^>]+)?>|https?://\S+")
DISCORD_URL_BRACKET_RE = re.compile(r"<(https?://[^>]+)>")

def _norm_text(t: str) -> str:
    if not isinstance(t, str):
        return ""
    # unify Discord/Slack artifacts
    t = DISCORD_URL_BRACKET_RE.sub(r"\1", t)
    t = URL_RE.sub("<URL>", t)
    t = MENTION_RE.sub("@user", t)
    t = t.replace("\r", " ").replace("\n", " ").strip()
    return t

def _to_datetime(ts: str) -> datetime:
    # Slack "1754260003.240174" (epoch.s), Discord ISO8601
    if ts and ts.replace(".", "", 1).isdigit():
        return datetime.utcfromtimestamp(float(ts))
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))

@dataclass
class WindowConfig:
    history_max_turns: int = 6            # context turns before assistant reply
    min_turns_required: int = 1           # require at least 1 prior turn
    dedup_blanks: bool = True             # drop empty messages


# ------------------------
# Slack loaders / preprocess
# ------------------------

def load_slack_jsonl(path: str) -> Dataset:
    """Load Slack messages from JSONL (recommended)."""
    ds = load_dataset("json", data_files=path, split="train",
                      json_kwargs={"lines": True})
    # convert to pandas once for grouping convenience
    return Dataset.from_pandas(pd.DataFrame(ds))

def preprocess_slack_to_chat(ds: Dataset, target_user: str, win: WindowConfig) -> Dataset:
    """Produce chat examples with `system` + `messages` (your original format)."""
    df = pd.DataFrame(ds)
    # keep human messages only
    df = df[(df.get("message_type", "message") == "message") & (~df.get("hidden", False))]
    df["thread_key"] = df.apply(
        lambda r: (r.get("channel_id"), r.get("thread_ts") or r.get("ts")), axis=1
    )
    df["ts_dt"] = df["ts"].apply(_to_datetime)
    df["text"] = df["text"].map(_norm_text)
    df = df.sort_values(["thread_key", "ts_dt"])

    def thread_examples(g: pd.DataFrame) -> List[Dict]:
        convo = []
        out = []
        for _, r in g.iterrows():
            role = "assistant" if r.get("user_name") == target_user else "user"
            if win.dedup_blanks and not r["text"]:
                continue
            convo.append({"role": role, "content": r["text"]})
            # whenever assistant speaks, create one training example
            if role == "assistant":
                ctx = convo[:-1][-win.history_max_turns:]
                if len(ctx) >= win.min_turns_required:
                    out.append({
                        "system": f"Write in the tone of @{target_user}",
                        "messages": ctx + [{"role": "assistant", "content": convo[-1]["content"]}],
                    })
        return out

    examples = list(itertools.chain.from_iterable(
        thread_examples(g) for _, g in df.groupby("thread_key", sort=False)
        if (g.get("user_name") == target_user).any()
    ))
    return Dataset.from_list(examples).train_test_split(test_size=0.1, seed=7)


# ------------------------
# Discord loaders / preprocess
# ------------------------

def load_discord_json(path: str) -> Dataset:
    """Load Discord export JSON (single file with 'messages' array)."""
    with open(path) as f:
        raw = json.load(f)
    msgs = raw["messages"]
    rows = []
    for m in msgs:
        content = _norm_text(m.get("content", ""))
        rows.append({
            "guild_id": raw.get("guild", {}).get("id"),
            "channel_id": raw.get("channel", {}).get("id"),
            "channel_name": raw.get("channel", {}).get("name"),
            "ts": m.get("timestamp"),
            "author_id": m.get("author", {}).get("id"),
            "author_name": m.get("author", {}).get("name"),
            "is_bot": m.get("author", {}).get("isBot", False),
            "content": content,
            "type": m.get("type", "Default"),
            "has_attachments": bool(m.get("attachments"))
        })
    return Dataset.from_pandas(pd.DataFrame(rows))

def preprocess_discord_to_chat(ds: Dataset, target_user: str, win: WindowConfig) -> Dataset:
    """Map Discord linear channel history to chat examples."""
    df = pd.DataFrame(ds)
    # keep user messages only
    df = df[(df.get("type", "Default") == "Default") & (~df.get("is_bot", False))]
    df["ts_dt"] = df["ts"].apply(_to_datetime)
    df["text"] = df["content"].map(_norm_text)
    df = df.sort_values(["channel_id", "ts_dt"])

    def channel_examples(g: pd.DataFrame) -> List[Dict]:
        convo = []
        out = []
        for _, r in g.iterrows():
            role = "assistant" if r.get("author_name") == target_user or r.get("author_id") == target_user else "user"
            if win.dedup_blanks and not r["text"]:
                continue
            convo.append({"role": role, "content": r["text"]})
            if role == "assistant":
                ctx = convo[:-1][-win.history_max_turns:]
                if len(ctx) >= win.min_turns_required:
                    out.append({
                        "system": f"Write in the tone of @{target_user}",
                        "messages": ctx + [{"role": "assistant", "content": convo[-1]["content"]}],
                    })
        return out

    examples = list(itertools.chain.from_iterable(
        channel_examples(g) for _, g in df.groupby("channel_id", sort=False)
        if (g.get("author_name") == target_user).any() or (g.get("author_id") == target_user).any()
    ))
    return Dataset.from_list(examples).train_test_split(test_size=0.1, seed=7)


# ------------------------
# Convert chat -> Alpaca-style SFT
# ------------------------

def chat_to_alpaca(ds: Dataset, name_field_for_headers: Optional[str] = None) -> Dataset:
    """
    Turn {"system", "messages":[...]} into Alpaca-style triples.
    instruction := prior turns rendered as 'speaker: text' lines (+ optional system header).
    output := final assistant message.
    """
    def render_instruction(sample: Dict) -> Tuple[str, str]:
        sys = sample.get("system") or ""
        msgs = sample["messages"]
        assert msgs[-1]["role"] == "assistant"
        ctx = msgs[:-1]
        # human-readable context
        lines = []
        if sys:
            lines.append(f"[SYSTEM] {sys}")
        for m in ctx:
            speaker = "User" if m["role"] != "assistant" else "Assistant"
            lines.append(f"{speaker}: {m['content']}")
        instruction = "\n".join(lines).strip()
        output = msgs[-1]["content"]
        return instruction, output

    def mapper(batch):
        inst, outp = [], []
        for m in batch["messages"]:
            instruction, output = render_instruction({"messages": m, "system": batch.get("system", [""] * len(batch["messages"]))[0] if isinstance(batch.get("system"), list) else batch.get("system")})
            inst.append(instruction)
            outp.append(output)
        return {"instruction": inst, "input": ["" for _ in inst], "output": outp}

    cols = ds["train"].column_names if isinstance(ds, dict) else ds.column_names
    mapped = ds.map(mapper, batched=True, remove_columns=cols)
    return mapped


# ------------------------
# Tokenization for chat-mode fine-tuning (unchanged behavior)
# ------------------------

def tokenize_dataset_chat(dataset: Dataset, tok: PreTrainedTokenizer, max_length: int = 4096) -> Dataset:
    """Tokenize chat-style dataset with label masking on the prompt."""
    if tok.chat_template is None:
        tok.chat_template = "{{ bos_token }}{% if system %}<|system|>\n{{ system }}\n{% endif %}{% for m in messages %}<|{{m.role}}|>\n{{ m.content }}\n{% endfor %}<|assistant|>\n"

    def to_features(batch):
        prompts = [
            tok.apply_chat_template(
                ([{"role": "system", "content": s}] if s else [])
                + b[:-1] + [{"role": "assistant", "content": ""}],
                tokenize=False,
            )
            for b, s in zip(batch["messages"], batch.get("system", [""] * len(batch["messages"])))
        ]
        targets = [b[-1]["content"] for b in batch["messages"]]
        full = [p + t for p, t in zip(prompts, targets)]
        enc_full = tok(full, max_length=max_length, truncation=True, padding=True)
        enc_prompt = tok(prompts, max_length=max_length, truncation=True, padding=True, add_special_tokens=False)
        labels = []
        for ids, pids in zip(enc_full["input_ids"], enc_prompt["input_ids"]):
            P = len(pids)
            lab = [-100] * P + ids[P:]
            lab = lab + [-100] * (len(ids) - len(lab))
            labels.append(lab)
        enc_full["labels"] = labels
        return enc_full

    return dataset.map(to_features, batched=True, remove_columns=dataset["train"].column_names)


# ------------------------
# High-level helpers you can call from your CLI
# ------------------------

def build_chat_dataset(source: str, path: str, target_user: str, win: WindowConfig = WindowConfig()) -> Dataset:
    source = source.lower()
    if source == "slack":
        ds = load_slack_jsonl(path)
        return preprocess_slack_to_chat(ds, target_user, win)
    elif source == "discord":
        ds = load_discord_json(path)
        return preprocess_discord_to_chat(ds, target_user, win)
    else:
        raise ValueError("source must be 'slack' or 'discord'")

def build_alpaca_dataset(source: str, path: str, target_user: str, win: WindowConfig = WindowConfig()) -> Dataset:
    chat = build_chat_dataset(source, path, target_user, win)
    return chat_to_alpaca(chat)
