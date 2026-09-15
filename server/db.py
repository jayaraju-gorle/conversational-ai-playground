"""SQLite database initialization and schema for the Conversational AI Playground.

Creates and manages the playground.db database with tables for:
- Tenant accounts, sessions, and per-tenant agent settings
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

CURRENT_SCHEMA_VERSION = 2


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

            -- Tenants
            CREATE TABLE IF NOT EXISTS tenants (
                id TEXT PRIMARY KEY,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                org_name TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS sessions (
                token_hash TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                expires_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tenant_settings (
                tenant_id TEXT PRIMARY KEY REFERENCES tenants(id) ON DELETE CASCADE,
                org_name TEXT DEFAULT '',
                default_language TEXT DEFAULT 'en-IN',
                default_llm TEXT DEFAULT '',
                default_stt TEXT DEFAULT 'sarvam',
                default_tts TEXT DEFAULT 'sarvam',
                default_voice TEXT DEFAULT 'aditya',
                lab_enabled INTEGER DEFAULT 0,
                temperature REAL DEFAULT 0.7,
                max_tokens INTEGER DEFAULT 1024,
                top_k INTEGER DEFAULT 5,
                custom_system_prompt TEXT DEFAULT '',
                use_custom_prompt INTEGER DEFAULT 0,
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS tenant_scenarios (
                tenant_id TEXT NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
                scenario_key TEXT NOT NULL,
                persona TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (tenant_id, scenario_key)
            );

            -- Knowledge Bases
            CREATE TABLE IF NOT EXISTS knowledge_bases (
                id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL DEFAULT '',
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
                tenant_id TEXT NOT NULL DEFAULT '',
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

        _migrate_schema(conn)

        # Set schema version if not set
        row = conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()
        if row[0] == 0:
            conn.execute("INSERT INTO schema_version (version) VALUES (?)",
                         (CURRENT_SCHEMA_VERSION,))
        else:
            conn.execute("UPDATE schema_version SET version = ?", (CURRENT_SCHEMA_VERSION,))

        conn.commit()
    finally:
        conn.close()


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Additive migrations for existing playground.db files."""
    conv_cols = _table_columns(conn, "conversations")
    if "client_id" not in conv_cols:
        conn.execute("ALTER TABLE conversations ADD COLUMN client_id TEXT DEFAULT ''")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversations_client ON conversations(client_id)"
        )
    if "tenant_id" not in conv_cols:
        conn.execute("ALTER TABLE conversations ADD COLUMN tenant_id TEXT NOT NULL DEFAULT ''")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversations_tenant ON conversations(tenant_id)"
        )

    kb_cols = _table_columns(conn, "knowledge_bases")
    if "tenant_id" not in kb_cols:
        conn.execute("ALTER TABLE knowledge_bases ADD COLUMN tenant_id TEXT NOT NULL DEFAULT ''")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_kb_tenant ON knowledge_bases(tenant_id)"
        )

    conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_tenant ON sessions(tenant_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at)")

    # Existing rows (pre-tenancy) stay attached to an unusable system tenant
    # so they are not visible to newly registered accounts.
    orphan_kbs = conn.execute(
        "SELECT COUNT(*) FROM knowledge_bases WHERE tenant_id = ''"
    ).fetchone()[0]
    orphan_convs = conn.execute(
        "SELECT COUNT(*) FROM conversations WHERE tenant_id = ''"
    ).fetchone()[0]
    if orphan_kbs or orphan_convs:
        existing = conn.execute(
            "SELECT id FROM tenants WHERE id = 'migrated'"
        ).fetchone()
        if not existing:
            conn.execute(
                """INSERT INTO tenants (id, email, password_hash, org_name)
                   VALUES ('migrated', 'migrated@local', 'unusable', 'Migrated data')"""
            )
        if orphan_kbs:
            conn.execute(
                "UPDATE knowledge_bases SET tenant_id = 'migrated' WHERE tenant_id = ''"
            )
        if orphan_convs:
            conn.execute(
                "UPDATE conversations SET tenant_id = 'migrated' WHERE tenant_id = ''"
            )


# Auto-initialize on import
init_db()
