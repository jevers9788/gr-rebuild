# Generate a realistic synthetic Slack conversation dataset and save as CSV and JSONL

import json
import random
import string
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

random.seed(42)


# Helpers
def slack_ts(dt):
    # Slack "ts" style: seconds.microseconds as string
    return f"{int(dt.timestamp())}.{random.randint(0, 999999):06d}"


def rid(prefix, n=8):
    return prefix + "".join(random.choices(string.ascii_uppercase + string.digits, k=n))


# Basic entities
team_id = "T" + "".join(random.choices(string.ascii_uppercase + string.digits, k=8))
workspaces = [{"team_id": team_id, "name": "gently-regressive-mock"}]

users = [
    {
        "user_id": "U01AAA",
        "user_name": "alice",
        "real_name": "Alice Chen",
        "role": "admin",
        "is_bot": False,
    },
    {
        "user_id": "U02BBB",
        "user_name": "bob",
        "real_name": "Bob Singh",
        "role": "member",
        "is_bot": False,
    },
    {
        "user_id": "U03CCC",
        "user_name": "carol",
        "real_name": "Carol Diaz",
        "role": "member",
        "is_bot": False,
    },
    {
        "user_id": "U04DDD",
        "user_name": "daphne",
        "real_name": "Daphne Lee",
        "role": "member",
        "is_bot": False,
    },
    {
        "user_id": "U05EEE",
        "user_name": "ed",
        "real_name": "Ed Zimmerman",
        "role": "owner",
        "is_bot": False,
    },
]

channels = [
    {
        "channel_id": "C01ENG",
        "channel_name": "eng-infra",
        "channel_type": "public_channel",
        "is_private": False,
    },
    {
        "channel_id": "C02PRD",
        "channel_name": "product",
        "channel_type": "private_channel",
        "is_private": True,
    },
    {
        "channel_id": "C03RAN",
        "channel_name": "random",
        "channel_type": "public_channel",
        "is_private": False,
    },
]

# Direct messages (im) and multi-party IMs (mpim)
ims = [
    {
        "channel_id": "D01ALB",
        "channel_name": "dm-alice-bob",
        "channel_type": "im",
        "is_private": True,
    },
]
mpims = [
    {
        "channel_id": "G01ABCD",
        "channel_name": "mpim-alice-bob-carol",
        "channel_type": "mpim",
        "is_private": True,
    },
]

all_channels = channels + ims + mpims

start = datetime.now(timezone.utc) - timedelta(days=10)
current = start

texts_pool = [
    "Morning folks — deploy at 10:30 ET. Any blockers?",
    "No blockers here. I’ll prep the migration script.",
    "Reminder: please update to Python 3.12 by Friday.",
    "Can we squeeze in the search relevance tweak?",
    "Logging 500s spiked after the last rollout; investigating.",
    "Great job on the latency reduction!",
    "Design doc draft is ready: <https://example.com/design/search|Search Revamp>",
    "We need a decision on index sharding by tomorrow.",
    "I'll take the oncall this weekend.",
    "FYI: security patch CVE-2025-12345 lands today.",
    "Re: feature flags — let's default off for 5% of users.",
    "Ship it. 🚀",
    "Can we move this to a thread?",
    "LGTM",
    "nit: rename var to request_id",
    "This is failing on ARM runners only.",
    "I'll pair with you after lunch.",
    "Docs PR: <https://github.com/acme/docs/pull/42>",
    "Meeting notes are in the wiki.",
    "Handoff complete; pager duty acknowledged.",
]

code_snippet = (
    "```python\ndef search(query: str) -> list:\n    return engine.lookup(query)\n```"
)
multiline = "Multi-line message:\n• point one\n• point two\n\nThanks!"

reactions_list = [
    [{"name": "thumbsup", "count": 3, "users": ["U01AAA", "U02BBB", "U03CCC"]}],
    [{"name": "rocket", "count": 2, "users": ["U02BBB", "U03CCC"]}],
    [{"name": "eyes", "count": 1, "users": ["U04DDD"]}],
    [],
]

files_pool = [
    {
        "id": rid("F"),
        "name": "sprint_plan.xlsx",
        "mimetype": "application/vnd.ms-excel",
    },
    {"id": rid("F"), "name": "error_log.txt", "mimetype": "text/plain"},
    {"id": rid("F"), "name": "diagram.png", "mimetype": "image/png"},
]


def random_text():
    t = random.choice(texts_pool + [code_snippet, multiline])
    # sprinkle mentions sometimes
    if random.random() < 0.25:
        mention = random.choice(users)["user_id"]
        t = f"<@{mention}> " + t
    return t


rows = []


def add_message(
    channel,
    user,
    ts=None,
    text=None,
    thread_ts=None,
    subtype=None,
    edited=False,
    hidden=False,
    bot_profile=None,
):
    global rows
    dt = ts if isinstance(ts, datetime) else current
    s_ts = slack_ts(dt if isinstance(dt, datetime) else current)
    message = {
        "team_id": team_id,
        "channel_id": channel["channel_id"],
        "channel_name": channel["channel_name"],
        "channel_type": channel["channel_type"],
        "is_private": channel["is_private"],
        "ts": s_ts,
        "thread_ts": thread_ts if thread_ts else s_ts,
        "is_thread_reply": thread_ts is not None,
        "user_id": user["user_id"] if user else None,
        "user_name": user["user_name"] if user else None,
        "user_role": user["role"] if user else None,
        "text": text if text else random_text(),
        "reactions": random.choice(reactions_list),
        "mentions": [u["user_id"] for u in random.sample(users, k=random.randint(0, 2))]
        if random.random() < 0.2
        else [],
        "files": random.sample(files_pool, k=random.randint(0, 1))
        if random.random() < 0.15
        else [],
        "edited": edited,
        "hidden": hidden,
        "subtype": subtype,
        "bot_profile": bot_profile,
        "reply_count": 0,
        "reply_users": [],
        "reply_users_count": 0,
        "app_id": None,
        "client_msg_id": rid("M"),
        "language": "en",
        "message_type": "message" if not subtype else "event",
    }
    rows.append(message)
    return message


# Create messages across channels, with some threads
current = start
for ch in all_channels:
    # 8 root messages per channel
    for _i in range(8):
        current += timedelta(minutes=random.randint(5, 90))
        u = random.choice([u for u in users if not u["is_bot"]])
        root = add_message(ch, u, ts=current)
        # 0-4 replies
        for _r in range(random.randint(0, 4)):
            current += timedelta(minutes=random.randint(1, 30))
            u2 = random.choice([u for u in users if not u["is_bot"]])
            reply = add_message(ch, u2, ts=current, thread_ts=root["thread_ts"])
            # increment reply stats on root
            root["reply_count"] += 1
            if u2["user_id"] not in root["reply_users"]:
                root["reply_users"].append(u2["user_id"])
                root["reply_users_count"] = len(root["reply_users"])

# Add some special cases
# Edited message
current += timedelta(minutes=5)
edited_msg = add_message(
    channels[0],
    users[0],
    ts=current,
    text="Deploy scheduled for 11:00 ET (was 10:30).",
    edited=True,
)

# Deleted/tombstoned
current += timedelta(minutes=3)
deleted_msg = add_message(
    channels[1],
    users[1],
    ts=current,
    text="This message was removed by the author.",
    subtype="message_deleted",
    hidden=True,
)


# Slash-command like message (represented as bot or app message)
current += timedelta(minutes=2)
slash_msg = add_message(
    channels[2],
    users[2],
    ts=current,
    text="/remind me to rotate keys next Monday at 9am",
)

# Build DataFrame
df = pd.DataFrame(rows)


# Clean up lists/dicts for CSV by JSON-encoding
def encode_json_cols(df, cols):
    for c in cols:
        df[c] = df[c].apply(lambda v: json.dumps(v, ensure_ascii=False))
    return df


json_cols = ["reactions", "mentions", "files", "reply_users", "bot_profile"]
csv_df = encode_json_cols(df.copy(), json_cols)

# Sort by timestamp
csv_df = csv_df.sort_values(by=["ts"]).reset_index(drop=True)
df = df.loc[csv_df.index]  # keep same order

# Save files
out_dir = Path(__file__).parent.parent.resolve() / "data"
csv_path = out_dir / "slack_messages_sample.csv"
jsonl_path = out_dir / "slack_messages_sample.jsonl"
readme_path = out_dir / "slack_messages_README.md"

csv_df.to_csv(csv_path, index=False)

with open(jsonl_path, "w", encoding="utf-8") as f:
    for rec in df.to_dict(orient="records"):
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

schema_md = """# Synthetic Slack Conversation Dataset (Sample)

This dataset simulates Slack exports across public/private channels, DMs, MPIMs, threads, edits, deletions, bots, attachments, code blocks, mentions, and reactions. It is synthetic and safe to share.

## Files
- `slack_messages_sample.csv` — spreadsheet-friendly; nested fields JSON-encoded.
- `slack_messages_sample.jsonl` — one JSON object per line.
- This README.

## Schema (per message)
- `team_id` (str): Workspace/team id.
- `channel_id` (str), `channel_name` (str), `channel_type` (enum: public_channel | private_channel | im | mpim), `is_private` (bool).
- `ts` (str): Slack timestamp `"seconds.microseconds"`.
- `thread_ts` (str): Root message ts for the thread; equals `ts` for root messages.
- `is_thread_reply` (bool).
- `user_id` (str|null), `user_name` (str|null), `user_role` (str|null), `language` (str).
- `text` (str): May include markdown, mentions `<@Uxxxx>`, links `<https://...|label>`, code blocks.
- `reactions` (list of {name, count, users}).
- `mentions` (list of user_ids).
- `files` (list of {id, name, mimetype}).
- `edited` (bool): Simulates user-edited messages.
- `hidden` (bool): True for deleted/tombstoned events.
- `subtype` (str|null): e.g., `message_deleted`, `bot_message`.
- `bot_profile` (obj|null): Present for bot/app messages.
- `reply_count` (int), `reply_users` (list), `reply_users_count` (int).
- `app_id` (str|null), `client_msg_id` (str): Synthetic client id.
- `message_type` (str): `"message"` for normal user posts, `"event"` for special subtypes.

## Notes
- Timestamps are within the last ~10 days.
- Mentions, reactions, and files are randomly populated to cover edge cases.
- You can group by `channel_id` and `thread_ts` to reconstruct threads.
"""

with open(readme_path, "w", encoding="utf-8") as f:
    f.write(schema_md)

# Show a concise preview
preview_cols = [
    "ts",
    "channel_name",
    "channel_type",
    "user_name",
    "text",
    "thread_ts",
    "is_thread_reply",
    "reactions",
]
preview = csv_df[preview_cols].head(20)
