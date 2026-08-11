"""Conversation history persistence for the Conversational AI Playground.

Stores and retrieves conversation sessions with full config snapshots,
messages, and metrics for later review and comparison.
"""

import json
import uuid

from loguru import logger

from db import get_connection


def save_conversation(data: dict) -> dict:
    """Save a conversation session.

    Expected data fields:
        title, scenario, llm_provider, llm_model, stt_provider, tts_provider,
        voice, language, knowledge_base_ids, mode, total_cost_usd, total_tokens,
        prompt_tokens, completion_tokens, avg_latency_ms, duration_seconds,
        messages (list of {role, content, timestamp, ...}),
        config_snapshot (dict of full config at session time)
    """
    conv_id = uuid.uuid4().hex[:12]

    # Auto-generate title from first user message if not provided
    title = data.get("title", "").strip()
    if not title:
        messages = data.get("messages", [])
        for msg in messages:
            if msg.get("role") == "user" and msg.get("content"):
                title = msg["content"][:80]
                if len(msg["content"]) > 80:
                    title += "…"
                break
        if not title:
            title = f"{data.get('scenario', 'generic')} conversation"

    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO conversations
               (id, title, scenario, llm_provider, llm_model, stt_provider,
                tts_provider, voice, language, knowledge_base_ids, mode,
                total_cost_usd, total_tokens, prompt_tokens, completion_tokens,
                avg_latency_ms, duration_seconds, messages, config_snapshot, client_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                conv_id,
                title,
                data.get("scenario", "generic"),
                data.get("llm_provider", ""),
                data.get("llm_model", ""),
                data.get("stt_provider", ""),
                data.get("tts_provider", ""),
                data.get("voice", ""),
                data.get("language", "en-IN"),
                json.dumps(data.get("knowledge_base_ids", [])),
                data.get("mode", "text"),
                data.get("total_cost_usd", 0.0),
                data.get("total_tokens", 0),
                data.get("prompt_tokens", 0),
                data.get("completion_tokens", 0),
                data.get("avg_latency_ms", 0.0),
                data.get("duration_seconds", 0.0),
                json.dumps(data.get("messages", [])),
                json.dumps(data.get("config_snapshot", {})),
                data.get("client_id", ""),
            ),
        )
        conn.commit()
        logger.info(f"Saved conversation '{title}' (id={conv_id}, client_id={data.get('client_id', '')})")
        return {"id": conv_id, "title": title}
    finally:
        conn.close()


def list_conversations(
    scenario: str | None = None,
    llm_provider: str | None = None,
    mode: str | None = None,
    client_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    """List conversations with optional filters and pagination.

    Returns {conversations: [...], total: int}.
    """
    conn = get_connection()
    try:
        conditions = []
        params: list = []

        if client_id:
            conditions.append("client_id = ?")
            params.append(client_id)
        if scenario:
            conditions.append("scenario = ?")
            params.append(scenario)
        if llm_provider:
            conditions.append("llm_provider = ?")
            params.append(llm_provider)
        if mode:
            conditions.append("mode = ?")
            params.append(mode)

        where = " WHERE " + " AND ".join(conditions) if conditions else ""

        # Total count
        total = conn.execute(
            f"SELECT COUNT(*) FROM conversations{where}", params
        ).fetchone()[0]

        # Paginated results (exclude messages blob for listing efficiency)
        rows = conn.execute(
            f"""SELECT id, created_at, title, scenario, llm_provider, llm_model,
                       stt_provider, tts_provider, voice, language,
                       knowledge_base_ids, mode, total_cost_usd, total_tokens,
                       prompt_tokens, completion_tokens, avg_latency_ms,
                       duration_seconds
                FROM conversations{where}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?""",
            params + [limit, offset],
        ).fetchall()

        conversations = []
        for r in rows:
            d = dict(r)
            d["knowledge_base_ids"] = json.loads(d.get("knowledge_base_ids") or "[]")
            conversations.append(d)

        return {"conversations": conversations, "total": total}
    finally:
        conn.close()


def get_conversation(conv_id: str) -> dict | None:
    """Get a single conversation with full messages."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM conversations WHERE id = ?", (conv_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["messages"] = json.loads(d.get("messages") or "[]")
        d["knowledge_base_ids"] = json.loads(d.get("knowledge_base_ids") or "[]")
        d["config_snapshot"] = json.loads(d.get("config_snapshot") or "{}")
        return d
    finally:
        conn.close()


def delete_conversation(conv_id: str) -> bool:
    """Delete a conversation."""
    conn = get_connection()
    try:
        cursor = conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
        conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info(f"Deleted conversation id={conv_id}")
        return deleted
    finally:
        conn.close()
