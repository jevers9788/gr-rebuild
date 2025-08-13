import itertools
import re

import pandas as pd
from datasets import Dataset
from transformers import PreTrainedTokenizer


def load_slack_messages(path: str) -> Dataset:
    """Load Slack messages from a CSV file."""
    df = pd.read_csv(path)
    return Dataset.from_pandas(df)


def preprocess_slack_messages(dataset: Dataset, target_user: str) -> Dataset:
    """Preprocess Slack messages from a dataset."""
    df = pd.read_json(dataset.path, lines=True)
    df = df[(df.message_type == "message") & (~df.hidden)]
    df = df.sort_values(["channel_id", "thread_ts", "ts"])

    def norm(t):
        t = re.sub(r"<https?://[^>|]+(\|[^>]+)?>", "<URL>", t)
        return t.strip()

    def thread_examples(g, target_user: str):
        convo = []
        out = []
        for _, r in g.iterrows():
            role = "assistant" if r["user_name"] == target_user else "user"
            convo.append({"role": role, "content": norm(r["text"])})
            if role == "assistant":
                ctx = [
                    {"role": "user", "content": m["content"]}
                    if m["role"] != "assistant"
                    else {"role": "assistant", "content": m["content"]}
                    for m in convo[:-1]
                ]
                out.append(
                    {
                        "system": "Write in the tone of @" + target_user,
                        "messages": ctx
                        + [{"role": "assistant", "content": convo[-1]["content"]}],
                    }
                )
        return out

    groups = df.groupby(["channel_id", "thread_ts"], sort=False)
    examples = list(
        itertools.chain.from_iterable(
            thread_examples(g, target_user)
            for _, g in groups
            if (g.user_name == target_user).any()
        )
    )
    ds = Dataset.from_list(examples).train_test_split(test_size=0.1, seed=7)
    return ds


def tokenize_dataset(dataset: Dataset, tok: PreTrainedTokenizer) -> Dataset:
    """Tokenize a dataset."""
    if tok.chat_template is None:
        tok.chat_template = "{{ bos_token }}{% if system %}<|system|>\n{{ system }}\n{% endif %}{% for m in messages %}<|{{m.role}}|>\n{{ m.content }}\n{% endfor %}<|assistant|>\n"

    def to_features(batch):
        prompts = [
            tok.apply_chat_template(
                [{"role": "system", "content": b["system"]}]
                + b["messages"][:-1]
                + [{"role": "assistant", "content": ""}],
                tokenize=False,
            )
            for b in batch["messages"]
        ]
        targets = [b["messages"][-1]["content"] for b in batch["messages"]]
        full = [p + t for p, t in zip(prompts, targets)]
        enc_full = tok(full, max_length=4096, truncation=True, padding=True)
        enc_prompt = tok(
            prompts,
            max_length=4096,
            truncation=True,
            padding=True,
            add_special_tokens=False,
        )
        labels = []
        for ids, pids in zip(enc_full["input_ids"], enc_prompt["input_ids"]):
            P = len(pids)
            lab = [-100] * P + ids[P:]
            lab = lab + [-100] * (len(ids) - len(lab))
            labels.append(lab)
        enc_full["labels"] = labels
        return enc_full

    tokenized = dataset.map(
        to_features, batched=True, remove_columns=dataset["train"].column_names
    )
    return tokenized
