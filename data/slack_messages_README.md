# Synthetic Slack Conversation Dataset (Sample)

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
