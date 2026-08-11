"""SQLite database initialization and schema for the Conversational AI Playground.

Creates and manages the playground.db database with tables for:
- Knowledge bases and their document chunks (with embeddings)
- Conversation history with full config snapshots and metrics

The DB file is auto-created at server/data/playground.db on first import.
"""

import json
import os
import sqlite3
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DB_PATH = DATA_DIR / "playground.db"
KB_FILES_DIR = DATA_DIR / "knowledge_bases"

CURRENT_SCHEMA_VERSION = 1


def _ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    KB_FILES_DIR.mkdir(parents=True, exist_ok=True)


def get_connection() -> sqlite3.Connection:
    """Get a SQLite connection with WAL mode and foreign keys enabled."""
    _ensure_dirs()
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create all tables if they don't exist."""
    _ensure_dirs()
    conn = get_connection()
    try:
        conn.executescript("""
            -- Schema versioning
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER NOT NULL
            );

            -- Knowledge Bases
            CREATE TABLE IF NOT EXISTS knowledge_bases (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            -- Documents uploaded to a KB
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                kb_id TEXT NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
                filename TEXT NOT NULL,
                file_size INTEGER DEFAULT 0,
                content_type TEXT DEFAULT 'text/plain',
                chunk_count INTEGER DEFAULT 0,
                status TEXT DEFAULT 'processing',
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                error_message TEXT
            );

            -- Chunked text with embeddings for RAG retrieval
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                kb_id TEXT NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding TEXT,
                token_count INTEGER DEFAULT 0
            );

            -- Conversation history
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                title TEXT DEFAULT '',
                scenario TEXT DEFAULT 'generic',
                llm_provider TEXT DEFAULT '',
                llm_model TEXT DEFAULT '',
                stt_provider TEXT DEFAULT '',
                tts_provider TEXT DEFAULT '',
                voice TEXT DEFAULT '',
                language TEXT DEFAULT 'en-IN',
                knowledge_base_ids TEXT DEFAULT '[]',
                mode TEXT DEFAULT 'text',
                total_cost_usd REAL DEFAULT 0.0,
                total_tokens INTEGER DEFAULT 0,
                prompt_tokens INTEGER DEFAULT 0,
                completion_tokens INTEGER DEFAULT 0,
                avg_latency_ms REAL DEFAULT 0.0,
                duration_seconds REAL DEFAULT 0.0,
                messages TEXT DEFAULT '[]',
                config_snapshot TEXT DEFAULT '{}'
            );

            -- Migration check for client_id column
            CREATE INDEX IF NOT EXISTS idx_conversations_created ON conversations(created_at);
            CREATE INDEX IF NOT EXISTS idx_conversations_scenario ON conversations(scenario);
            CREATE INDEX IF NOT EXISTS idx_conversations_llm ON conversations(llm_provider);
        """)

        # Migration: ensure client_id column exists on conversations table
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(conversations)").fetchall()]
        if "client_id" not in cols:
            conn.execute("ALTER TABLE conversations ADD COLUMN client_id TEXT DEFAULT ''")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_client ON conversations(client_id)")

        # Set schema version if not set
        row = conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()
        if row[0] == 0:
            conn.execute("INSERT INTO schema_version (version) VALUES (?)",
                         (CURRENT_SCHEMA_VERSION,))

        conn.commit()
    finally:
        conn.close()


# Auto-initialize on import
init_db()
