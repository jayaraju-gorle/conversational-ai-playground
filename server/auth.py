"""Tenant accounts, sessions, and per-tenant agent settings."""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import uuid
from datetime import datetime, timedelta, timezone

from loguru import logger

from db import get_connection

SESSION_COOKIE = "pg_session"
SESSION_DAYS = 14
PBKDF2_ROUNDS = 120_000


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), PBKDF2_ROUNDS
    )
    return f"pbkdf2_sha256${PBKDF2_ROUNDS}${salt}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _algo, rounds, salt, digest = stored.split("$", 3)
        dk = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt.encode("utf-8"), int(rounds)
        )
        return hmac.compare_digest(dk.hex(), digest)
    except Exception:
        return False


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _row_tenant(row) -> dict:
    return {
        "id": row["id"],
        "email": row["email"],
        "org_name": row["org_name"],
        "created_at": row["created_at"],
    }


def create_tenant(email: str, password: str, org_name: str) -> dict:
    email = email.strip().lower()
    org_name = org_name.strip()
    tenant_id = uuid.uuid4().hex[:12]
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO tenants (id, email, password_hash, org_name)
               VALUES (?, ?, ?, ?)""",
            (tenant_id, email, hash_password(password), org_name),
        )
        conn.execute(
            """INSERT INTO tenant_settings (tenant_id, org_name)
               VALUES (?, ?)""",
            (tenant_id, org_name),
        )
        conn.commit()
        logger.info(f"Created tenant '{org_name}' ({email}, id={tenant_id})")
        return {"id": tenant_id, "email": email, "org_name": org_name}
    finally:
        conn.close()


def authenticate(email: str, password: str) -> dict | None:
    email = email.strip().lower()
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM tenants WHERE email = ?", (email,)
        ).fetchone()
        if not row or not verify_password(password, row["password_hash"]):
            return None
        return _row_tenant(row)
    finally:
        conn.close()


def get_tenant(tenant_id: str) -> dict | None:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM tenants WHERE id = ?", (tenant_id,)
        ).fetchone()
        return _row_tenant(row) if row else None
    finally:
        conn.close()


def email_taken(email: str) -> bool:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT 1 FROM tenants WHERE email = ?", (email.strip().lower(),)
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def create_session(tenant_id: str) -> str:
    token = secrets.token_urlsafe(32)
    expires = _now() + timedelta(days=SESSION_DAYS)
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO sessions (token_hash, tenant_id, expires_at)
               VALUES (?, ?, ?)""",
            (hash_token(token), tenant_id, _iso(expires)),
        )
        conn.commit()
        return token
    finally:
        conn.close()


def tenant_from_session(token: str | None) -> dict | None:
    if not token:
        return None
    conn = get_connection()
    try:
        row = conn.execute(
            """SELECT t.id, t.email, t.org_name, t.created_at, s.expires_at
               FROM sessions s
               JOIN tenants t ON t.id = s.tenant_id
               WHERE s.token_hash = ?""",
            (hash_token(token),),
        ).fetchone()
        if not row:
            return None
        expires = datetime.fromisoformat(row["expires_at"])
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        if expires < _now():
            conn.execute(
                "DELETE FROM sessions WHERE token_hash = ?", (hash_token(token),)
            )
            conn.commit()
            return None
        return {
            "id": row["id"],
            "email": row["email"],
            "org_name": row["org_name"],
            "created_at": row["created_at"],
        }
    finally:
        conn.close()


def delete_session(token: str | None) -> None:
    if not token:
        return
    conn = get_connection()
    try:
        conn.execute(
            "DELETE FROM sessions WHERE token_hash = ?", (hash_token(token),)
        )
        conn.commit()
    finally:
        conn.close()


def default_settings(org_name: str = "") -> dict:
    return {
        "org_name": org_name,
        "default_language": "en-IN",
        "default_llm": os.getenv("LLM_PROVIDER") or "",
        "default_stt": os.getenv("STT_PROVIDER") or "sarvam",
        "default_tts": os.getenv("TTS_PROVIDER") or "sarvam",
        "default_voice": "aditya",
        "lab_enabled": False,
        "temperature": 0.7,
        "max_tokens": 1024,
        "top_k": 5,
        "custom_system_prompt": "",
        "use_custom_prompt": False,
    }


def get_settings(tenant_id: str) -> dict:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM tenant_settings WHERE tenant_id = ?", (tenant_id,)
        ).fetchone()
        base = default_settings()
        if not row:
            return base
        d = dict(row)
        base.update(
            {
                "org_name": d.get("org_name") or base["org_name"],
                "default_language": d.get("default_language") or "en-IN",
                "default_llm": d.get("default_llm") or "",
                "default_stt": d.get("default_stt") or "sarvam",
                "default_tts": d.get("default_tts") or "sarvam",
                "default_voice": d.get("default_voice") or "aditya",
                "lab_enabled": bool(d.get("lab_enabled")),
                "temperature": float(d.get("temperature") or 0.7),
                "max_tokens": int(d.get("max_tokens") or 1024),
                "top_k": int(d.get("top_k") or 5),
                "custom_system_prompt": d.get("custom_system_prompt") or "",
                "use_custom_prompt": bool(d.get("use_custom_prompt")),
            }
        )
        return base
    finally:
        conn.close()


_SETTINGS_FIELDS = {
    "org_name",
    "default_language",
    "default_llm",
    "default_stt",
    "default_tts",
    "default_voice",
    "lab_enabled",
    "temperature",
    "max_tokens",
    "top_k",
    "custom_system_prompt",
    "use_custom_prompt",
}


def update_settings(tenant_id: str, patch: dict) -> dict:
    current = get_settings(tenant_id)
    for key, value in patch.items():
        if key not in _SETTINGS_FIELDS or value is None:
            continue
        if key == "lab_enabled" or key == "use_custom_prompt":
            current[key] = bool(value)
        elif key in ("temperature",):
            current[key] = float(value)
        elif key in ("max_tokens", "top_k"):
            current[key] = int(value)
        else:
            current[key] = value
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO tenant_settings (
                    tenant_id, org_name, default_language, default_llm, default_stt,
                    default_tts, default_voice, lab_enabled, temperature, max_tokens,
                    top_k, custom_system_prompt, use_custom_prompt, updated_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
               ON CONFLICT(tenant_id) DO UPDATE SET
                    org_name = excluded.org_name,
                    default_language = excluded.default_language,
                    default_llm = excluded.default_llm,
                    default_stt = excluded.default_stt,
                    default_tts = excluded.default_tts,
                    default_voice = excluded.default_voice,
                    lab_enabled = excluded.lab_enabled,
                    temperature = excluded.temperature,
                    max_tokens = excluded.max_tokens,
                    top_k = excluded.top_k,
                    custom_system_prompt = excluded.custom_system_prompt,
                    use_custom_prompt = excluded.use_custom_prompt,
                    updated_at = datetime('now')""",
            (
                tenant_id,
                current["org_name"],
                current["default_language"],
                current["default_llm"],
                current["default_stt"],
                current["default_tts"],
                current["default_voice"],
                1 if current["lab_enabled"] else 0,
                current["temperature"],
                current["max_tokens"],
                current["top_k"],
                current["custom_system_prompt"],
                1 if current["use_custom_prompt"] else 0,
            ),
        )
        if current["org_name"]:
            conn.execute(
                "UPDATE tenants SET org_name = ? WHERE id = ?",
                (current["org_name"], tenant_id),
            )
        conn.commit()
        return current
    finally:
        conn.close()


def get_scenario_persona(tenant_id: str, key: str) -> str | None:
    conn = get_connection()
    try:
        row = conn.execute(
            """SELECT persona FROM tenant_scenarios
               WHERE tenant_id = ? AND scenario_key = ?""",
            (tenant_id, key),
        ).fetchone()
        return row["persona"] if row else None
    finally:
        conn.close()


def list_scenario_personas(tenant_id: str) -> dict[str, str]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT scenario_key, persona FROM tenant_scenarios WHERE tenant_id = ?",
            (tenant_id,),
        ).fetchall()
        return {r["scenario_key"]: r["persona"] for r in rows}
    finally:
        conn.close()


def upsert_scenario_persona(tenant_id: str, key: str, persona: str) -> None:
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO tenant_scenarios (tenant_id, scenario_key, persona, updated_at)
               VALUES (?, ?, ?, datetime('now'))
               ON CONFLICT(tenant_id, scenario_key) DO UPDATE SET
                    persona = excluded.persona,
                    updated_at = datetime('now')""",
            (tenant_id, key, persona),
        )
        conn.commit()
    finally:
        conn.close()


def public_tenant(tenant: dict) -> dict:
    settings = get_settings(tenant["id"])
    return {
        "id": tenant["id"],
        "email": tenant["email"],
        "org_name": settings["org_name"] or tenant["org_name"],
        **{k: settings[k] for k in _SETTINGS_FIELDS if k != "org_name"},
    }


DEMO_TENANT_EMAIL = os.getenv("DEMO_TENANT_EMAIL", "demo@playground.ai").strip().lower()
DEMO_TENANT_PASSWORD = os.getenv("DEMO_TENANT_PASSWORD", "demo12345")
DEMO_TENANT_ORG = os.getenv("DEMO_TENANT_ORG", "Demo Workspace")


def ensure_demo_tenant() -> dict:
    """Ensure the default demo tenant exists for playground exploration."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM tenants WHERE email = ?", (DEMO_TENANT_EMAIL,)
        ).fetchone()
        if row:
            return _row_tenant(row)
    finally:
        conn.close()

    tenant = create_tenant(DEMO_TENANT_EMAIL, DEMO_TENANT_PASSWORD, DEMO_TENANT_ORG)
    try:
        import knowledge_base as kb_mod
        kb_mod.seed_sample_kbs(tenant["id"])
    except Exception as e:
        logger.warning(f"Could not seed sample KBs for demo tenant: {e}")
    return tenant
