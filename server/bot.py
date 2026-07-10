#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""AI Transformation Playground - Pipecat Voice Agent

Cascade pipeline: Speech-to-Text → LLM → Text-to-Speech, with
client-selectable providers, language, and voice. Serves a playground UI
that shows real cost / latency / token metrics per session.

Run the bot using::

    uv run bot.py
"""

import os
import time

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.azure.llm import AzureLLMService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.google.vertex.llm import GoogleVertexLLMService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.groq.stt import GroqSTTService
from pipecat.services.groq.tts import GroqTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.realtime import events as openai_realtime_events
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
from pipecat.services.openai.responses.llm import OpenAIResponsesLLMService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.sarvam.llm import SarvamLLMService
from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.services.sarvam.tts import SarvamTTSService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport, TransportParams

try:
    from pipecat.transports.daily.transport import DailyParams
    HAS_DAILY = True
except ImportError:
    HAS_DAILY = False
    DailyParams = None
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)


# ══════════════════════════════════════════════════════════════════════════
# Playground catalog: providers, languages, voices, pricing
# Served to the client via GET /api/config so the UI is data-driven.
# ══════════════════════════════════════════════════════════════════════════

# NOTE: rates are USD *estimates* for demo purposes — verify against each
# vendor's current pricing page before quoting customers.
PRICING = {
    # "billing": how audio reaches the provider. "stream" = continuous
    # websocket (silence is processed too — cost accrues with elapsed time);
    # "speech" = VAD-segmented (only user utterances are sent/billed).
    "stt": {
        "sarvam": {"per_min": 0.006, "billing": "stream"},     # Saaras ~₹30/hr
        "deepgram": {"per_min": 0.0077, "billing": "stream"},  # Nova streaming
        # Groq Whisper bills per audio hour: v3 $0.111/hr, turbo $0.04/hr
        "groq:whisper-large-v3": {"per_min": 0.00185, "billing": "speech"},
        "groq:whisper-large-v3-turbo": {"per_min": 0.000667, "billing": "speech"},
        # gpt-4o-transcribe ≈ $0.006/min audio; VAD-segmented like Whisper
        "azure:gpt-4o-transcribe": {"per_min": 0.006, "billing": "speech"},
    },
    "tts": {
        "sarvam": {"per_1k_chars": 0.018},  # Sarvam Bulbul
        "cartesia": {"per_1k_chars": 0.05}, # Cartesia Sonic
        "deepgram": {"per_1k_chars": 0.030},  # Deepgram Aura-2
        "groq": {"per_1k_chars": 0.022},    # Canopy Labs Orpheus $22/1M chars
    },
    "llm": {
        "google": {"in_per_1m": 0.30, "out_per_1m": 2.50},   # Gemini 3.5 Flash
        # Sarvam lists ₹2.5/₹10 (30B) and ₹4/₹16 (105B) per 1M — USD at ~₹88
        "sarvam:sarvam-30b": {"in_per_1m": 0.028, "out_per_1m": 0.114},
        "sarvam:sarvam-105b": {"in_per_1m": 0.045, "out_per_1m": 0.182},
        "openai": {"in_per_1m": 2.00, "out_per_1m": 8.00},   # GPT-4.1
        # Anthropic (Claude) list prices
        "anthropic:claude-opus-4-8": {"in_per_1m": 5.00, "out_per_1m": 25.00},
        # Sonnet 5 intro pricing ($2/$10) runs through 2026-08-31; list is $3/$15
        "anthropic:claude-sonnet-5": {"in_per_1m": 2.00, "out_per_1m": 10.00},
        "anthropic:claude-haiku-4-5": {"in_per_1m": 1.00, "out_per_1m": 5.00},
        # Groq production models
        "groq:llama-3.1-8b-instant": {"in_per_1m": 0.05, "out_per_1m": 0.08},
        "groq:llama-3.3-70b-versatile": {"in_per_1m": 0.59, "out_per_1m": 0.79},
        "groq:openai/gpt-oss-120b": {"in_per_1m": 0.15, "out_per_1m": 0.60},
        "groq:openai/gpt-oss-20b": {"in_per_1m": 0.075, "out_per_1m": 0.30},
        # Groq compound systems: no per-token list price — Groq bills the
        # built-in tools (search, code exec) separately, so cost shows $0 here
        "groq:groq/compound": {"in_per_1m": 0.0, "out_per_1m": 0.0},
        "groq:groq/compound-mini": {"in_per_1m": 0.0, "out_per_1m": 0.0},
        # Groq preview models
        "groq:meta-llama/llama-4-scout-17b-16e-instruct": {"in_per_1m": 0.11, "out_per_1m": 0.34},
        "groq:qwen/qwen3-32b": {"in_per_1m": 0.29, "out_per_1m": 0.59},
        "groq:qwen/qwen3.6-27b": {"in_per_1m": 0.60, "out_per_1m": 3.00},
        # Azure OpenAI (regional list prices mirror OpenAI's)
        "azure:gpt-5-mini": {"in_per_1m": 0.25, "out_per_1m": 2.00},
        # GCP Vertex AI (same list prices as the Gemini API)
        "vertex:gemini-3.5-flash": {"in_per_1m": 0.30, "out_per_1m": 2.50},
        "vertex:gemini-2.5-pro": {"in_per_1m": 1.25, "out_per_1m": 10.00},
        # xAI Grok on Vertex MaaS (xAI list prices)
        "vertex:xai/grok-4.1-fast-non-reasoning": {"in_per_1m": 0.20, "out_per_1m": 0.50},
        "vertex:xai/grok-4.3": {"in_per_1m": 3.00, "out_per_1m": 15.00},
        # Realtime S2S bills audio as tokens (Gemini Live native audio)
        "gemini-live": {"in_per_1m": 2.10, "out_per_1m": 8.50},
        # OpenAI gpt-realtime audio-token pricing
        "openai-realtime": {"in_per_1m": 32.00, "out_per_1m": 64.00},
    },
}

USD_INR_RATE = float(os.getenv("USD_INR_RATE", "88.0"))

# NOTE: actual coverage depends on the selected STT/TTS: Sarvam covers the
# Indian set; Deepgram Nova and Groq Whisper cover most of the international
# set. The bot always instructs the LLM to reply in the chosen language.
LANGUAGES = [
    {"code": "en-IN", "label": "English (India)", "group": "Indian"},
    {"code": "hi-IN", "label": "Hindi", "group": "Indian"},
    {"code": "bn-IN", "label": "Bengali", "group": "Indian"},
    {"code": "gu-IN", "label": "Gujarati", "group": "Indian"},
    {"code": "kn-IN", "label": "Kannada", "group": "Indian"},
    {"code": "ml-IN", "label": "Malayalam", "group": "Indian"},
    {"code": "mr-IN", "label": "Marathi", "group": "Indian"},
    {"code": "or-IN", "label": "Odia", "group": "Indian"},
    {"code": "pa-IN", "label": "Punjabi", "group": "Indian"},
    {"code": "ta-IN", "label": "Tamil", "group": "Indian"},
    {"code": "te-IN", "label": "Telugu", "group": "Indian"},
    {"code": "ur-IN", "label": "Urdu", "group": "Indian"},
    {"code": "en-US", "label": "English (US)", "group": "International"},
    {"code": "en-GB", "label": "English (UK)", "group": "International"},
    {"code": "es-ES", "label": "Spanish", "group": "International"},
    {"code": "fr-FR", "label": "French", "group": "International"},
    {"code": "de-DE", "label": "German", "group": "International"},
    {"code": "it-IT", "label": "Italian", "group": "International"},
    {"code": "pt-BR", "label": "Portuguese (Brazil)", "group": "International"},
    {"code": "nl-NL", "label": "Dutch", "group": "International"},
    {"code": "pl-PL", "label": "Polish", "group": "International"},
    {"code": "ru-RU", "label": "Russian", "group": "International"},
    {"code": "tr-TR", "label": "Turkish", "group": "International"},
    {"code": "ar-SA", "label": "Arabic", "group": "International"},
    {"code": "zh-CN", "label": "Chinese (Mandarin)", "group": "International"},
    {"code": "ja-JP", "label": "Japanese", "group": "International"},
    {"code": "ko-KR", "label": "Korean", "group": "International"},
    {"code": "id-ID", "label": "Indonesian", "group": "International"},
    {"code": "vi-VN", "label": "Vietnamese", "group": "International"},
    {"code": "th-TH", "label": "Thai", "group": "International"},
]

# Language-support sets referenced by provider entries below. The client
# shows only languages every active provider supports (missing key = all).
_INDIAN_LANGS = [lang["code"] for lang in LANGUAGES if lang["group"] == "Indian"]
# Bulbul TTS covers the Indian set except Urdu (Saaras STT does support Urdu)
_SARVAM_TTS_LANGS = [c for c in _INDIAN_LANGS if c != "ur-IN"]
_INTL_LANGS = [lang["code"] for lang in LANGUAGES if lang["group"] == "International"]
_ALL_LANGS = [lang["code"] for lang in LANGUAGES]
_ENGLISH_LANGS = ["en-IN", "en-US", "en-GB"]
# Cartesia Sonic's multilingual set
_CARTESIA_LANGS = _ENGLISH_LANGS + [
    "hi-IN", "es-ES", "fr-FR", "de-DE", "it-IT", "pt-BR", "nl-NL", "pl-PL",
    "ru-RU", "tr-TR", "zh-CN", "ja-JP", "ko-KR",
]
# Deepgram Nova: broad international coverage + Hindi (no regional Indic)
_DEEPGRAM_LANGS = _INTL_LANGS + ["en-IN", "hi-IN"]

SARVAM_V3_VOICES = [
    "aditya", "ritu", "priya", "neha", "rahul", "pooja", "rohan", "simran",
    "kavya", "amit", "dev", "ishita", "shreya", "ratan", "varun", "manan",
    "sumit", "roopa", "kabir", "aayan", "shubh", "ashutosh", "advait",
    "amelia", "sophia",
]

# Multi-model providers encode the model in the value as "provider:model-id";
# the server splits it back apart (see split_provider) so the client can stay
# a dumb pass-through. "group" renders as an <optgroup> in the dropdown, and
# "requires" lists the env vars an entry needs — /api/config turns that into
# an "available" flag so unconfigured entries show up disabled, not broken.
PROVIDERS = {
    "llm": [
        # ── Direct provider APIs ──
        {"value": "google", "label": "Gemini 3.5 Flash", "group": "Direct",
         "requires": ["GEMINI_API_KEY|GOOGLE_API_KEY"]},
        {"value": "openai", "label": "GPT-4.1", "group": "Direct",
         "requires": ["OPENAI_API_KEY"]},
        {"value": "sarvam:sarvam-30b", "label": "Sarvam 30B", "group": "Direct",
         "requires": ["SARVAM_API_KEY"]},
        {"value": "sarvam:sarvam-105b", "label": "Sarvam 105B", "group": "Direct",
         "requires": ["SARVAM_API_KEY"]},
        {"value": "anthropic:claude-opus-4-8", "label": "Claude Opus 4.8", "group": "Direct",
         "requires": ["ANTHROPIC_API_KEY"]},
        {"value": "anthropic:claude-sonnet-5", "label": "Claude Sonnet 5", "group": "Direct",
         "requires": ["ANTHROPIC_API_KEY"]},
        {"value": "anthropic:claude-haiku-4-5", "label": "Claude Haiku 4.5", "group": "Direct",
         "requires": ["ANTHROPIC_API_KEY"]},
        # ── Hosted on Groq ──
        {"value": "groq:llama-3.3-70b-versatile", "label": "Llama 3.3 70B",
         "group": "Hosted on Groq", "requires": ["GROQ_API_KEY"]},
        {"value": "groq:llama-3.1-8b-instant", "label": "Llama 3.1 8B Instant",
         "group": "Hosted on Groq", "requires": ["GROQ_API_KEY"]},
        {"value": "groq:openai/gpt-oss-120b", "label": "OpenAI GPT-OSS 120B",
         "group": "Hosted on Groq", "requires": ["GROQ_API_KEY"]},
        {"value": "groq:openai/gpt-oss-20b", "label": "OpenAI GPT-OSS 20B",
         "group": "Hosted on Groq", "requires": ["GROQ_API_KEY"]},
        {"value": "groq:groq/compound", "label": "Groq Compound",
         "group": "Hosted on Groq", "requires": ["GROQ_API_KEY"]},
        {"value": "groq:groq/compound-mini", "label": "Groq Compound Mini",
         "group": "Hosted on Groq", "requires": ["GROQ_API_KEY"]},
        {"value": "groq:meta-llama/llama-4-scout-17b-16e-instruct",
         "label": "Llama 4 Scout 17B (preview)",
         "group": "Hosted on Groq", "requires": ["GROQ_API_KEY"]},
        {"value": "groq:qwen/qwen3-32b", "label": "Qwen3 32B (preview)",
         "group": "Hosted on Groq", "requires": ["GROQ_API_KEY"]},
        {"value": "groq:qwen/qwen3.6-27b", "label": "Qwen3.6 27B (preview)",
         "group": "Hosted on Groq", "requires": ["GROQ_API_KEY"]},
        # ── Hosted on Azure OpenAI (model = your deployment name; run
        # /api/models/sync or check the Foundry portal to see deployments) ──
        {"value": "azure:gpt-5-mini", "label": "OpenAI GPT-5 mini",
         "group": "Hosted on Azure",
         "requires": ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"]},
        # ── Hosted on GCP Vertex AI ──
        {"value": "vertex:gemini-3.5-flash", "label": "Gemini 3.5 Flash",
         "group": "Hosted on GCP Vertex",
         "requires": ["GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_VERTEX_PROJECT_ID"]},
        {"value": "vertex:gemini-2.5-pro", "label": "Gemini 2.5 Pro",
         "group": "Hosted on GCP Vertex",
         "requires": ["GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_VERTEX_PROJECT_ID"]},
        # xAI Grok via Vertex Model-as-a-Service (OpenAI-compatible endpoint,
        # global location only; requires Model Garden terms acceptance)
        {"value": "vertex:xai/grok-4.1-fast-non-reasoning", "label": "Grok 4.1 Fast",
         "group": "Hosted on GCP Vertex",
         "requires": ["GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_VERTEX_PROJECT_ID"]},
        {"value": "vertex:xai/grok-4.3", "label": "Grok 4.3",
         "group": "Hosted on GCP Vertex",
         "requires": ["GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_VERTEX_PROJECT_ID"]},
        # ── Realtime: speech-to-speech — replaces the whole STT/LLM/TTS cascade ──
        {"value": "gemini-live", "label": "Gemini Live (S2S)", "realtime": True,
         "group": "Realtime (S2S)", "requires": ["GEMINI_API_KEY|GOOGLE_API_KEY"],
         "languages": _ALL_LANGS},
        {"value": "openai-realtime", "label": "OpenAI Realtime (S2S)", "realtime": True,
         "group": "Realtime (S2S)", "requires": ["OPENAI_API_KEY"],
         "languages": _ALL_LANGS},
    ],
    "stt": [
        {"value": "sarvam", "label": "Sarvam Saaras v3", "requires": ["SARVAM_API_KEY"],
         "languages": _INDIAN_LANGS},
        {"value": "deepgram", "label": "Deepgram Nova", "requires": ["DEEPGRAM_API_KEY"],
         "languages": _DEEPGRAM_LANGS},
        {"value": "groq:whisper-large-v3", "label": "Whisper v3 (Groq)",
         "requires": ["GROQ_API_KEY"], "languages": _ALL_LANGS},
        {"value": "groq:whisper-large-v3-turbo", "label": "Whisper v3 Turbo (Groq)",
         "requires": ["GROQ_API_KEY"], "languages": _ALL_LANGS},
        {"value": "azure:gpt-4o-transcribe", "label": "GPT-4o Transcribe (Azure)",
         "requires": ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"],
         "languages": _ALL_LANGS},
    ],
    "tts": [
        {"value": "sarvam", "label": "Sarvam Bulbul v3", "requires": ["SARVAM_API_KEY"],
         "languages": _SARVAM_TTS_LANGS},
        {"value": "cartesia", "label": "Cartesia Sonic", "requires": ["CARTESIA_API_KEY"],
         "languages": _CARTESIA_LANGS},
        {"value": "deepgram", "label": "Deepgram Aura-2 (English)",
         "requires": ["DEEPGRAM_API_KEY"], "languages": _ENGLISH_LANGS},
        {"value": "groq", "label": "Groq Orpheus (English only)", "requires": ["GROQ_API_KEY"],
         "languages": _ENGLISH_LANGS},
    ],
}


def _vertex_access_token() -> str:
    """Short-lived OAuth token for Vertex MaaS (OpenAI-compatible) calls."""
    from google.auth.transport.requests import Request
    from google.oauth2 import service_account

    creds = service_account.Credentials.from_service_account_file(
        os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    creds.refresh(Request())
    return creds.token


def _vertex_maas_base() -> str:
    """OpenAI-compatible base URL for Vertex partner models (global only)."""
    proj = os.getenv("GOOGLE_VERTEX_PROJECT_ID")
    return f"https://aiplatform.googleapis.com/v1/projects/{proj}/locations/global/endpoints/openapi"


def _azure_endpoint() -> str:
    """AZURE_OPENAI_ENDPOINT normalized to scheme://host.

    The Azure/Foundry portals surface full API URLs (e.g. .../openai/v1/responses);
    pasting one of those must not break the paths we build on top.
    """
    from urllib.parse import urlparse

    raw = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    return f"{parsed.scheme}://{parsed.netloc}"


def _env_set(name: str) -> bool:
    """True if the env var (or any |-separated alternative) is non-empty."""
    return any(os.getenv(alt) for alt in name.split("|"))


def providers_with_availability() -> dict:
    """PROVIDERS with an "available" flag resolved from the current env."""
    out = {}
    for kind, entries in PROVIDERS.items():
        out[kind] = [
            {**e, "available": all(_env_set(r) for r in e.get("requires", []))}
            for e in entries
        ]
    return out

GEMINI_LIVE_VOICES = ["Puck", "Charon", "Kore", "Fenrir", "Aoede", "Leda", "Orus", "Zephyr"]

OPENAI_REALTIME_VOICES = [
    "marin", "cedar", "alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse",
]

# Canopy Labs Orpheus v1 English voices on Groq
GROQ_ORPHEUS_VOICES = ["autumn", "diana", "hannah", "austin", "daniel", "troy"]

# Curated Deepgram Aura-2 voices (full catalog is ~40; IDs are aura-2-<name>-en)
DEEPGRAM_AURA_VOICES = [
    "aura-2-thalia-en", "aura-2-helena-en", "aura-2-andromeda-en", "aura-2-athena-en",
    "aura-2-luna-en", "aura-2-apollo-en", "aura-2-arcas-en", "aura-2-orion-en",
    "aura-2-zeus-en",
]

VOICES = {
    "sarvam": [{"value": v, "label": v.capitalize()} for v in SARVAM_V3_VOICES],
    "groq": [{"value": v, "label": v.capitalize()} for v in GROQ_ORPHEUS_VOICES],
    "deepgram": [
        {"value": v, "label": v.removeprefix("aura-2-").removesuffix("-en").capitalize()}
        for v in DEEPGRAM_AURA_VOICES
    ],
    # Keyed by the realtime LLM value: the client swaps the voice list to this
    # set when a realtime provider is selected (voice belongs to the S2S model).
    "gemini-live": [{"value": v, "label": v} for v in GEMINI_LIVE_VOICES],
    "openai-realtime": [{"value": v, "label": v.capitalize()} for v in OPENAI_REALTIME_VOICES],
    "cartesia": [
        {
            "value": os.getenv("CARTESIA_VOICE_ID", "71a7ad14-091c-4e8e-a314-022ece01c121"),
            "label": "Default Voice",
        }
    ],
}


# Scenario templates: each swaps the agent's persona so a prospect can try
# the bot as *their* business. Availability/account data is simulated — the
# persona says so explicitly, so the bot behaves like a demo, not a liar.
SCENARIOS = {
    "generic": {
        "label": "General Assistant",
        "persona": (
            "You are a helpful business assistant for customer-facing sales"
            " and support conversations."
        ),
        "greeting": "Start by concisely introducing yourself and asking how you can help.",
        "sample": [
            "Hi, who are you and what can you help me with?",
            "What kind of businesses do you usually assist?",
            "Alright, thanks. That's all for now.",
        ],
    },
    "hospital": {
        "label": "Hospital — Appointment Booking",
        "persona": (
            "You are the appointment booking assistant for Sunrise Multi-Speciality"
            " Hospital. Help callers book, reschedule, or cancel doctor appointments."
            " Collect one detail at a time: the patient's name, the specialty or"
            " doctor they need (General Medicine, Cardiology, Orthopedics, Pediatrics,"
            " Dermatology, ENT), and their preferred date and time. Consultations are"
            " available Monday to Saturday, 9 AM to 1 PM and 5 PM to 8 PM. This is a"
            " product demo, so simulate realistic slot availability — occasionally a"
            " requested slot is taken and you offer the nearest alternatives. Before"
            " confirming, read back the full booking details. Never give medical"
            " advice; for emergencies, tell the caller to visit the emergency"
            " department immediately."
        ),
        "greeting": (
            "Greet the caller as Sunrise Hospital's appointment assistant and ask"
            " how you can help."
        ),
        "sample": [
            "Hi, I'd like to book an appointment with a cardiologist.",
            "My name is Ravi Kumar.",
            "Tomorrow morning around 10 would be great.",
            "Yes, please confirm that booking.",
            "No, that's all. Thank you!",
        ],
    },
    "bank": {
        "label": "Bank — Customer Support",
        "persona": (
            "You are a customer support agent for Horizon Bank. Help customers with"
            " blocking a lost or stolen card, checking account balance, recent"
            " transactions, branch locations and timings, and general product"
            " questions about savings accounts, fixed deposits, and credit cards."
            " Before discussing any account-specific detail, verify identity by"
            " asking for the customer's registered name and the last four digits of"
            " their account number. This is a product demo, so simulate plausible"
            " account data after verification. Never ask for full card numbers,"
            " PINs, passwords, or OTPs — if a caller offers one, tell them never to"
            " share it. For a lost card, treat it as urgent: verify, confirm the"
            " card is blocked, and reassure the customer."
        ),
        "greeting": (
            "Greet the caller as Horizon Bank customer support and ask how you can"
            " help."
        ),
        "sample": [
            "Hi, I think I lost my debit card. I'm really worried.",
            "My name is Priya Sharma, account ending 4321.",
            "Yes, please block the card immediately.",
            "How do I get a replacement card?",
            "Thanks, that's very helpful. Bye.",
        ],
    },
    "ecommerce": {
        "label": "E-commerce — Order Support",
        "persona": (
            "You are the order support assistant for QuickKart, an online shopping"
            " platform. Help customers track orders, process returns and refunds,"
            " and answer questions about delivery times and payment issues. Ask for"
            " the order ID or the customer's registered phone number to look up an"
            " order. This is a product demo, so simulate plausible order details."
            " Returns are accepted within 7 days of delivery; refunds take 3 to 5"
            " business days. Be empathetic when a customer reports a problem, and"
            " summarize the resolution before ending."
        ),
        "greeting": (
            "Greet the caller as QuickKart order support and ask how you can help."
        ),
        "sample": [
            "Hi, my order hasn't arrived yet and it's been a week.",
            "The order ID is QK12345.",
            "It was supposed to arrive two days ago.",
            "If it doesn't arrive by then, can I get a refund?",
            "Okay, thank you for the help.",
        ],
    },
}


def resolve_scenario(key: str | None) -> dict:
    return SCENARIOS.get(key or "generic", SCENARIOS["generic"])


def split_provider(value: str | None) -> tuple[str | None, str | None]:
    """Split a "provider:model-id" selector into (provider, model).

    Plain values ("sarvam", "gemini-live") come back with model=None.
    """
    if not value:
        return None, None
    provider, _, model = value.partition(":")
    return provider, model or None


def resolve_language(code: str) -> tuple[Language, str]:
    """Map a client language code to a pipecat Language enum + display label."""
    label = next((l["label"] for l in LANGUAGES if l["code"] == code), "English")
    try:
        return Language(code), label
    except ValueError:
        logger.warning(f"Unknown language code '{code}', falling back to en-IN")
        return Language.EN_IN, "English"


def build_system_instruction(
    language_label: str, voice_mode: bool = True, scenario: dict | None = None
) -> str:
    base = (scenario or SCENARIOS["generic"])["persona"]
    if voice_mode:
        base += (
            " Your responses will be spoken aloud, so avoid emojis, bullet"
            " points, or other formatting that can't be spoken. Keep responses"
            " brief and conversational."
        )
    else:
        base += " Keep responses helpful and brief."
    # The conversation language is a pipeline setting (STT transcription +
    # TTS synthesis), not something the LLM can change by itself — agreeing
    # to switch mid-conversation would break transcription and speech.
    lang_name = language_label.split(" (")[0]
    base += (
        f" Conduct the entire conversation in {lang_name}, regardless of the"
        f" language the user writes or speaks in. If the user asks you to"
        f" switch to a different language, do not switch — explain that the"
        f" conversation language is set with the Language selector in the"
        f" toolbar, and changing it there restarts the conversation in the"
        f" new language."
    )
    return base


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments) -> None:
    """Run the voice bot for this session."""
    logger.info("Starting bot")

    # ── Read client selections from request body ──
    body = (runner_args.body if runner_args and runner_args.body else {}) or {}

    language_code = body.get("language") or "en-IN"
    language, language_label = resolve_language(language_code)
    logger.info(f"Using language: {language_code} ({language_label})")

    scenario = resolve_scenario(body.get("scenario"))
    logger.info(f"Using scenario: {scenario['label']}")

    # LLM provider resolves first: realtime (speech-to-speech) providers
    # replace the whole STT → LLM → TTS cascade with a single service.
    llm_selection = body.get("llm") or os.getenv("LLM_PROVIDER")
    if not llm_selection:
        if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            llm_selection = "google"
        elif os.getenv("SARVAM_API_KEY"):
            llm_selection = "sarvam"
        else:
            llm_selection = "openai"
    llm_provider, llm_model = split_provider(llm_selection)
    is_realtime = llm_provider in ("gemini-live", "openai-realtime")
    logger.info(f"Using LLM provider: {llm_provider} model={llm_model} (realtime={is_realtime})")

    system_instruction = build_system_instruction(
        language_label, voice_mode=True, scenario=scenario
    )

    stt = None
    tts = None
    if not is_realtime:
        # Speech-to-Text service
        stt_selection = body.get("stt") or os.getenv("STT_PROVIDER") or (
            "sarvam" if os.getenv("SARVAM_API_KEY") else "deepgram"
        )
        stt_provider, stt_model = split_provider(stt_selection)
        logger.info(f"Using STT provider: {stt_provider} model={stt_model}")

        if stt_provider == "sarvam":
            stt = SarvamSTTService(
                api_key=os.getenv("SARVAM_API_KEY"),
                settings=SarvamSTTService.Settings(
                    model="saaras:v3",
                    language=language,
                ),
            )
        elif stt_provider == "azure":
            # OpenAI-compatible transcription on the Azure v1 surface
            stt = OpenAISTTService(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                base_url=f"{_azure_endpoint()}/openai/v1",
                settings=OpenAISTTService.Settings(
                    model=stt_model or "gpt-4o-transcribe",
                    language=language,
                ),
            )
        elif stt_provider == "groq":
            stt = GroqSTTService(
                api_key=os.getenv("GROQ_API_KEY"),
                settings=GroqSTTService.Settings(
                    model=stt_model or "whisper-large-v3-turbo",
                    language=language,
                ),
            )
        else:
            stt = DeepgramSTTService(
                api_key=os.getenv("DEEPGRAM_API_KEY"),
                settings=DeepgramSTTService.Settings(language=language),
            )

        # Text-to-Speech service
        tts_provider = body.get("tts") or os.getenv("TTS_PROVIDER") or (
            "sarvam" if os.getenv("SARVAM_API_KEY") else "cartesia"
        )
        voice_selection = body.get("voice") or "aditya"
        logger.info(f"Using TTS provider: {tts_provider}, voice: {voice_selection}")

        if tts_provider == "sarvam":
            tts = SarvamTTSService(
                api_key=os.getenv("SARVAM_API_KEY"),
                settings=SarvamTTSService.Settings(
                    model="bulbul:v3-beta",
                    voice=voice_selection,
                    language=language,
                ),
            )
        elif tts_provider == "deepgram":
            dg_voice = (
                voice_selection if voice_selection in DEEPGRAM_AURA_VOICES
                else "aura-2-thalia-en"
            )
            tts = DeepgramTTSService(
                api_key=os.getenv("DEEPGRAM_API_KEY"),
                settings=DeepgramTTSService.Settings(voice=dg_voice),
            )
        elif tts_provider == "groq":
            groq_voice = voice_selection if voice_selection in GROQ_ORPHEUS_VOICES else "autumn"
            tts = GroqTTSService(
                api_key=os.getenv("GROQ_API_KEY"),
                settings=GroqTTSService.Settings(
                    model="canopylabs/orpheus-v1-english",
                    voice=groq_voice,
                ),
            )
        else:
            tts = CartesiaTTSService(
                api_key=os.getenv("CARTESIA_API_KEY"),
                settings=CartesiaTTSService.Settings(
                    voice=body.get("voice")
                    or os.getenv("CARTESIA_VOICE_ID", "71a7ad14-091c-4e8e-a314-022ece01c121"),
                ),
            )

    if llm_provider == "gemini-live":
        live_voice = body.get("voice") or "Puck"
        if live_voice not in GEMINI_LIVE_VOICES:
            live_voice = "Puck"
        logger.info(f"Using Gemini Live voice: {live_voice}")
        llm = GeminiLiveLLMService(
            api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
            settings=GeminiLiveLLMService.Settings(
                model=os.getenv(
                    "GEMINI_LIVE_MODEL",
                    "models/gemini-2.5-flash-native-audio-latest",
                ),
                voice=live_voice,
                language=language,
                system_instruction=system_instruction,
            ),
        )
    elif llm_provider == "openai-realtime":
        rt_voice = body.get("voice") or "marin"
        if rt_voice not in OPENAI_REALTIME_VOICES:
            rt_voice = "marin"
        logger.info(f"Using OpenAI Realtime voice: {rt_voice}")
        llm = OpenAIRealtimeLLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            settings=OpenAIRealtimeLLMService.Settings(
                model=os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-2"),
                system_instruction=system_instruction,
                session_properties=openai_realtime_events.SessionProperties(
                    audio=openai_realtime_events.AudioConfiguration(
                        output=openai_realtime_events.AudioOutput(voice=rt_voice),
                    ),
                ),
            ),
        )
    elif llm_provider == "anthropic":
        llm = AnthropicLLMService(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            settings=AnthropicLLMService.Settings(
                model=llm_model or "claude-opus-4-8",
                system_instruction=system_instruction,
            ),
        )
    elif llm_provider == "google":
        llm = GoogleLLMService(
            api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
            settings=GoogleLLMService.Settings(
                model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
                system_instruction=system_instruction,
            ),
        )
    elif llm_provider == "sarvam":
        llm = SarvamLLMService(
            api_key=os.getenv("SARVAM_API_KEY"),
            settings=SarvamLLMService.Settings(
                model=llm_model or os.getenv("SARVAM_LLM_MODEL", "sarvam-30b"),
                system_instruction=system_instruction,
            ),
        )
    elif llm_provider == "azure":
        llm = AzureLLMService(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            endpoint=_azure_endpoint(),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-09-01-preview"),
            settings=AzureLLMService.Settings(
                # Azure routes by deployment name; ours mirror the model names
                model=llm_model or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini"),
                system_instruction=system_instruction,
            ),
        )
    elif llm_provider == "vertex" and (llm_model or "").startswith("xai/"):
        # Partner models (Grok) speak OpenAI chat-completions on the MaaS
        # endpoint; auth is a short-lived OAuth token (fine per-session)
        llm = OpenAILLMService(
            api_key=_vertex_access_token(),
            base_url=_vertex_maas_base(),
            settings=OpenAILLMService.Settings(
                model=llm_model,
                system_instruction=system_instruction,
            ),
        )
    elif llm_provider == "vertex":
        llm = GoogleVertexLLMService(
            credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
            project_id=os.getenv("GOOGLE_VERTEX_PROJECT_ID"),
            # "global" routes to wherever the model is served — regional
            # locations (us-east4 etc.) don't carry every Gemini model
            location=os.getenv("GOOGLE_VERTEX_LOCATION", "global"),
            settings=GoogleVertexLLMService.Settings(
                model=llm_model or "gemini-3.5-flash",
                system_instruction=system_instruction,
            ),
        )
    elif llm_provider == "groq":
        groq_model = llm_model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        llm = GroqLLMService(
            api_key=os.getenv("GROQ_API_KEY"),
            settings=GroqLLMService.Settings(
                model=groq_model,
                system_instruction=system_instruction,
                # Qwen models emit <think> reasoning inline by default, which
                # TTS would speak aloud — ask Groq to strip it server-side.
                extra={"reasoning_format": "hidden"} if groq_model.startswith("qwen") else {},
            ),
        )
    else:
        llm = OpenAIResponsesLLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            settings=OpenAIResponsesLLMService.Settings(
                model=os.getenv("OPENAI_MODEL", "gpt-4.1"),
                system_instruction=system_instruction,
            ),
        )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    # Pipeline — cascade (STT → LLM → TTS) or realtime (S2S does both internally)
    if is_realtime:
        stages = [
            transport.input(),
            user_aggregator,
            llm,
            transport.output(),
            assistant_aggregator,
        ]
    else:
        stages = [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    pipeline = Pipeline(stages)

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[],
    )

    @worker.rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        # Kick off the conversation
        context.add_message({"role": "developer", "content": scenario["greeting"]})
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=False)

    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point."""

    from pipecat.evals.transport import EvalTransportParams

    transport_params = {
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
        # Headless eval transport: drive the same pipeline with scripted
        # scenarios (uv run bot.py -t eval), no live call needed.
        "eval": lambda: EvalTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    }

    if HAS_DAILY and DailyParams:
        transport_params["daily"] = lambda: DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        )

    transport = await create_transport(runner_args, transport_params)

    await run_bot(transport, runner_args)


if __name__ == "__main__":
    import pathlib

    import aiohttp
    from fastapi.responses import HTMLResponse
    from pipecat.runner.run import app, main
    from pydantic import BaseModel

    # ── TURN relay for WebRTC on hosts without inbound UDP (e.g. Cloud Run) ──
    # TURN_URLS is comma-separated, e.g.
    #   "turn:a.relay.metered.ca:443,turn:a.relay.metered.ca:443?transport=tcp"
    # Credentials go in TURN_USERNAME / TURN_PASSWORD. Without TURN_URLS,
    # behavior is unchanged (STUN only — fine for hosts with direct UDP).
    STUN_URL = "stun:stun.l.google.com:19302"
    TURN_URLS = [u.strip() for u in os.getenv("TURN_URLS", "").split(",") if u.strip()]

    def client_ice_servers() -> list[dict]:
        """ICE server list for the browser client (served via /api/config)."""
        servers: list[dict] = [{"urls": STUN_URL}]
        if TURN_URLS:
            servers.append(
                {
                    "urls": TURN_URLS,
                    "username": os.getenv("TURN_USERNAME", ""),
                    "credential": os.getenv("TURN_PASSWORD", ""),
                }
            )
        return servers

    # TURN_SERVER_SIDE=0 keeps TURN for the browser only: the bot then pairs its
    # own outbound UDP socket with the client's relay candidate, which works on
    # hosts with egress UDP (e.g. Cloud Run) and avoids aioice's TURN client.
    TURN_SERVER_SIDE = os.getenv("TURN_SERVER_SIDE", "1") not in ("0", "false")

    # aiortc only uses the first TURN URL, and its UDP TURN transactions have
    # proven flaky against some providers; TURN_SERVER_URLS lets the bot use a
    # different (e.g. TCP-only) URL than the browser.
    TURN_SERVER_URLS = [
        u.strip() for u in os.getenv("TURN_SERVER_URLS", "").split(",") if u.strip()
    ] or TURN_URLS

    if os.getenv("ICE_DEBUG"):
        import logging as _stdlog

        _stdlog.basicConfig(level=_stdlog.INFO)
        for _name in ("aioice", "aiortc"):
            _stdlog.getLogger(_name).setLevel(_stdlog.DEBUG)

    if TURN_URLS:
        # The dev runner builds its SmallWebRTCRequestHandler without exposing
        # an ice_servers hook, so inject them via the constructor. STUN is
        # always included: without it the bot only advertises its container IP,
        # and the client's TURN relay drops packets from the bot's (undisclosed)
        # public NAT address, so ICE never completes.
        from pipecat.transports.smallwebrtc import request_handler as _swrtc
        from pipecat.transports.smallwebrtc.connection import IceServer

        _server_ice_servers = [IceServer(urls=STUN_URL)]
        if TURN_SERVER_SIDE:
            _server_ice_servers.append(
                IceServer(
                    urls=TURN_SERVER_URLS,
                    username=os.getenv("TURN_USERNAME", ""),
                    credential=os.getenv("TURN_PASSWORD", ""),
                )
            )

        _orig_handler_init = _swrtc.SmallWebRTCRequestHandler.__init__

        def _handler_init_with_turn(self, *args, **kwargs):
            kwargs.setdefault("ice_servers", _server_ice_servers)
            _orig_handler_init(self, *args, **kwargs)

        _swrtc.SmallWebRTCRequestHandler.__init__ = _handler_init_with_turn

    # Override root to serve our custom playground page
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def root_playground():
        page = pathlib.Path(__file__).parent / "templates" / "client.html"
        return HTMLResponse(content=page.read_text(encoding="utf-8"))

    # Dynamically filter out /client routes on application startup
    @app.on_event("startup")
    async def remove_client_routes():
        app.router.routes = [
            r for r in app.router.routes if getattr(r, "path", None) not in ("/client", "/client/")
        ]

    # Live USD→INR rate from frankfurter.app (ECB reference rates; free, no
    # key), cached for 6 hours. USD_INR_RATE from .env is the offline fallback.
    _fx_cache = {"rate": None, "ts": 0.0}
    FX_TTL_SECONDS = 6 * 3600

    async def get_usd_inr_rate() -> tuple[float, str]:
        now = time.time()
        if _fx_cache["rate"] and now - _fx_cache["ts"] < FX_TTL_SECONDS:
            return _fx_cache["rate"], "live"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://api.frankfurter.app/latest?from=USD&to=INR",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        rate = float(data["rates"]["INR"])
                        _fx_cache.update(rate=rate, ts=now)
                        logger.info(f"Live USD->INR rate: {rate}")
                        return rate, "live"
                    logger.warning(f"FX API returned {resp.status}; using fallback rate")
        except Exception as e:
            logger.warning(f"FX fetch failed ({e}); using fallback rate")
        if _fx_cache["rate"]:  # stale live rate beats the static fallback
            return _fx_cache["rate"], "live-stale"
        return USD_INR_RATE, "fallback"

    @app.get("/api/config")
    async def playground_config():
        """Catalog the UI needs: providers, voices, languages, pricing, FX."""
        usd_inr, fx_source = await get_usd_inr_rate()
        return {
            "default_llm": os.getenv("LLM_PROVIDER") or None,
            "default_stt": os.getenv("STT_PROVIDER") or None,
            "default_tts": os.getenv("TTS_PROVIDER") or None,
            "providers": providers_with_availability(),
            "voices": VOICES,
            "languages": LANGUAGES,
            "scenarios": [
                {"value": k, "label": v["label"], "sample": v.get("sample", [])}
                for k, v in SCENARIOS.items()
            ],
            "pricing": PRICING,
            "usd_inr": usd_inr,
            "fx_source": fx_source,
            "ice_servers": client_ice_servers(),
        }

    # ── Model catalog sync check ─────────────────────────────────────────
    # Providers' model-list APIs tell us when the hand-curated PROVIDERS
    # catalog is stale (new models launched / catalog models retired). They
    # carry no pricing or display metadata, so this reports a diff for a
    # human to act on rather than mutating the catalog.

    def _catalog_models_for(provider: str) -> set[str]:
        """Model IDs the catalog references for a provider (LLM + STT values)."""
        ids = set()
        for kind in ("llm", "stt"):
            for entry in PROVIDERS[kind]:
                prov, model = split_provider(entry["value"])
                if prov == provider and model:
                    ids.add(model)
        return ids

    async def _fetch_json(session, url, headers=None):
        async with session.get(
            url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)
        ) as resp:
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status}: {(await resp.text())[:200]}")
            return await resp.json()

    async def _sync_groq(session) -> dict:
        data = await _fetch_json(
            session,
            "https://api.groq.com/openai/v1/models",
            {"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"},
        )
        live = {m["id"] for m in data["data"] if m.get("active", True)}
        known = _catalog_models_for("groq")
        known.add("canopylabs/orpheus-v1-english")  # Groq TTS (catalog value "groq")
        return {
            "live_count": len(live),
            "new": sorted(live - known),
            "missing_from_api": sorted(known - live),
        }

    async def _sync_anthropic(session) -> dict:
        data = await _fetch_json(
            session,
            "https://api.anthropic.com/v1/models?limit=100",
            {
                "x-api-key": os.getenv("ANTHROPIC_API_KEY"),
                "anthropic-version": "2023-06-01",
            },
        )
        live = {m["id"] for m in data["data"]}
        known = _catalog_models_for("anthropic")
        # Catalog uses aliases (claude-opus-4-8); the API lists dated snapshots
        # too, so only flag catalog entries with no live prefix match.
        missing = [k for k in known if not any(m.startswith(k) for m in live)]
        new = sorted(m for m in live if not any(m.startswith(k) for k in known))
        return {"live_count": len(live), "new": new, "missing_from_api": sorted(missing)}

    async def _sync_google(session) -> dict:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        data = await _fetch_json(
            session,
            f"https://generativelanguage.googleapis.com/v1beta/models?pageSize=200&key={api_key}",
        )
        live = {
            m["name"].removeprefix("models/")
            for m in data.get("models", [])
            if "generateContent" in m.get("supportedGenerationMethods", [])
        }
        known = {os.getenv("GEMINI_MODEL", "gemini-3.5-flash")}
        known |= _catalog_models_for("vertex")
        known.add(
            os.getenv("GEMINI_LIVE_MODEL", "models/gemini-2.5-flash-native-audio-latest")
            .removeprefix("models/")
        )
        # Gemini's list is long (previews, dated snapshots) — hide those from
        # "new" so the signal stays readable.
        new = sorted(
            m for m in live - known
            if "preview" not in m and "exp" not in m and not m[-1].isdigit()
        )
        return {
            "live_count": len(live),
            "new": new,
            "missing_from_api": sorted(known - live),
        }

    async def _sync_openai(session) -> dict:
        data = await _fetch_json(
            session,
            "https://api.openai.com/v1/models",
            {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"},
        )
        live = {m["id"] for m in data["data"]}
        known = {os.getenv("OPENAI_MODEL", "gpt-4.1")}
        known.add(os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-2"))
        known |= {split_provider(e["value"])[1] for e in PROVIDERS["llm"]
                  if e["value"].startswith("azure:")}
        # Chat-capable families only, skipping dated snapshots — the raw list
        # includes embeddings, whisper, tts, moderation, etc.
        import re
        new = sorted(
            m for m in live - known
            if re.match(r"^(gpt-|o\d|chatgpt)", m)
            and not re.search(r"\d{4}-\d{2}-\d{2}", m)
            and not any(t in m for t in ("embedding", "audio", "tts", "transcribe",
                                         "search", "moderation", "image", "instruct"))
        )
        return {
            "live_count": len(live),
            "new": new,
            "missing_from_api": sorted(known - live),
        }

    async def _sync_azure(session) -> dict:
        base = _azure_endpoint()
        if not base:
            raise RuntimeError("AZURE_OPENAI_ENDPOINT not set")
        data = await _fetch_json(
            session,
            f"{base}/openai/deployments?api-version=2023-03-15-preview",
            {"api-key": os.getenv("AZURE_OPENAI_API_KEY")},
        )
        live = {d["id"] for d in data["data"]}
        known = _catalog_models_for("azure")  # spans LLM + STT entries
        return {
            "live_count": len(live),
            "new": sorted(live - known),
            "missing_from_api": sorted(known - live),
        }

    @app.get("/api/models/sync")
    async def models_sync():
        """Diff the curated PROVIDERS catalog against each provider's live
        models API. Informational: 'new' = live models absent from the
        catalog (candidates to add, pricing/label needed); 'missing_from_api'
        = catalog entries the provider no longer lists (deprecation risk)."""
        checks = {
            "groq": ("GROQ_API_KEY", _sync_groq),
            "anthropic": ("ANTHROPIC_API_KEY", _sync_anthropic),
            "google": ("GEMINI_API_KEY|GOOGLE_API_KEY", _sync_google),
            "openai": ("OPENAI_API_KEY", _sync_openai),
            "azure": ("AZURE_OPENAI_API_KEY", _sync_azure),
        }
        results = {}
        async with aiohttp.ClientSession() as session:
            for name, (env, fn) in checks.items():
                if not _env_set(env):
                    results[name] = {"status": "skipped", "reason": f"{env} not set"}
                    continue
                try:
                    results[name] = {"status": "ok", **(await fn(session))}
                except Exception as e:
                    results[name] = {"status": "error", "reason": str(e)[:300]}
        for name in ("sarvam", "deepgram", "cartesia"):
            results[name] = {"status": "skipped", "reason": "no public models API"}
        return {
            "note": (
                "Model lists are curated by hand in bot.py (PROVIDERS/PRICING). "
                "'new' models need a price and label before adding; "
                "'missing_from_api' entries may be deprecated upstream."
            ),
            "providers": results,
        }

    class ChatMessage(BaseModel):
        role: str
        content: str

    class ChatRequest(BaseModel):
        message: str
        history: list[ChatMessage] = []
        llm: str = "google"
        language: str = "en-IN"
        scenario: str = "generic"

    def _chat_response(content, usage=None, latency_ms=None, error=False, model=None):
        return {
            "role": "assistant",
            "content": content,
            "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0},
            "latency_ms": latency_ms,
            "error": error,
            "model": model,
        }

    @app.post("/api/chat")
    async def chat_endpoint(request: ChatRequest):
        """Direct text chat against the selected LLM, with real usage + latency."""
        llm_provider, llm_model = split_provider(
            request.llm or os.getenv("LLM_PROVIDER") or "google"
        )
        # Realtime S2S providers have no text path — use the text sibling model
        if llm_provider == "gemini-live":
            llm_provider = "google"
        elif llm_provider == "openai-realtime":
            llm_provider = "openai"
        _, language_label = resolve_language(request.language)
        system_instruction = build_system_instruction(
            language_label, voice_mode=False, scenario=resolve_scenario(request.scenario)
        )

        t_start = time.monotonic()

        is_vertex_maas = llm_provider == "vertex" and (llm_model or "").startswith("xai/")
        if llm_provider in ("google", "vertex") and not is_vertex_maas:
            import google.genai as genai
            from google.genai import types

            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if llm_provider == "vertex":
                model_name = llm_model or "gemini-3.5-flash"
            else:
                model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

            contents = []
            for msg in request.history:
                role = "user" if msg.role == "user" else "model"
                contents.append(
                    types.Content(role=role, parts=[types.Part.from_text(text=msg.content)])
                )
            contents.append(
                types.Content(role="user", parts=[types.Part.from_text(text=request.message)])
            )

            try:
                if llm_provider == "vertex":
                    # Uses GOOGLE_APPLICATION_CREDENTIALS for auth
                    client = genai.Client(
                        vertexai=True,
                        project=os.getenv("GOOGLE_VERTEX_PROJECT_ID"),
                        location=os.getenv("GOOGLE_VERTEX_LOCATION", "global"),
                    )
                else:
                    client = genai.Client(api_key=api_key)
                response = await client.aio.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(system_instruction=system_instruction),
                )
                latency_ms = round((time.monotonic() - t_start) * 1000)
                usage = None
                if response.usage_metadata:
                    usage = {
                        "prompt_tokens": response.usage_metadata.prompt_token_count or 0,
                        "completion_tokens": response.usage_metadata.candidates_token_count or 0,
                    }
                return _chat_response(response.text, usage, latency_ms, model=model_name)
            except Exception as e:
                logger.error(f"Gemini API error: {e}")
                return _chat_response(f"Error calling Gemini API: {e}", error=True)

        if llm_provider == "anthropic":
            import anthropic as anthropic_sdk

            model_name = llm_model or "claude-opus-4-8"
            payload_messages = [
                {"role": m.role, "content": m.content} for m in request.history
            ]
            payload_messages.append({"role": "user", "content": request.message})
            try:
                anthropic_client = anthropic_sdk.AsyncAnthropic(
                    api_key=os.getenv("ANTHROPIC_API_KEY")
                )
                message = await anthropic_client.messages.create(
                    model=model_name,
                    max_tokens=2048,
                    system=system_instruction,
                    messages=payload_messages,
                )
                latency_ms = round((time.monotonic() - t_start) * 1000)
                if message.stop_reason == "refusal":
                    return _chat_response(
                        "Claude declined this request for safety reasons.",
                        error=True,
                    )
                content = "".join(b.text for b in message.content if b.type == "text")
                usage = {
                    "prompt_tokens": message.usage.input_tokens,
                    "completion_tokens": message.usage.output_tokens,
                }
                return _chat_response(content, usage, latency_ms, model=model_name)
            except anthropic_sdk.APIStatusError as e:
                return _chat_response(
                    f"Anthropic Error ({e.status_code}): {e.message}", error=True
                )
            except Exception as e:
                logger.error(f"Anthropic API error: {e}")
                return _chat_response(f"Error calling Anthropic API: {e}", error=True)

        # Sarvam, Groq, and OpenAI all speak the OpenAI chat-completions protocol
        if llm_provider == "sarvam":
            url = "https://api.sarvam.ai/v1/chat/completions"
            headers = {
                "api-subscription-key": os.getenv("SARVAM_API_KEY"),
                "Content-Type": "application/json",
            }
            model = llm_model or os.getenv("SARVAM_LLM_MODEL", "sarvam-30b")
            provider_name = "Sarvam AI"
        elif llm_provider == "groq":
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
                "Content-Type": "application/json",
            }
            model = llm_model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
            provider_name = "Groq"
        elif is_vertex_maas:
            url = f"{_vertex_maas_base()}/chat/completions"
            headers = {
                "Authorization": f"Bearer {_vertex_access_token()}",
                "Content-Type": "application/json",
            }
            model = llm_model
            provider_name = "Vertex (xAI)"
        elif llm_provider == "azure":
            model = llm_model or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini")
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-09-01-preview")
            url = (
                f"{_azure_endpoint()}/openai/deployments/{model}/chat/completions"
                f"?api-version={api_version}"
            )
            headers = {
                "api-key": os.getenv("AZURE_OPENAI_API_KEY"),
                "Content-Type": "application/json",
            }
            provider_name = "Azure OpenAI"
        else:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                "Content-Type": "application/json",
            }
            model = os.getenv("OPENAI_MODEL", "gpt-4.1")
            provider_name = "OpenAI"

        payload_messages = [{"role": "system", "content": system_instruction}]
        for msg in request.history:
            payload_messages.append({"role": msg.role, "content": msg.content})
        payload_messages.append({"role": "user", "content": request.message})

        payload = {"model": model, "messages": payload_messages}
        # Qwen on Groq emits <think> reasoning inline unless told to hide it
        if llm_provider == "groq" and model.startswith("qwen"):
            payload["reasoning_format"] = "hidden"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        latency_ms = round((time.monotonic() - t_start) * 1000)
                        content = data["choices"][0]["message"]["content"]
                        raw_usage = data.get("usage") or {}
                        usage = {
                            "prompt_tokens": raw_usage.get("prompt_tokens", 0),
                            "completion_tokens": raw_usage.get("completion_tokens", 0),
                        }
                        return _chat_response(content, usage, latency_ms, model=model)
                    text = await resp.text()
                    return _chat_response(
                        f"{provider_name} Error ({resp.status}): {text}", error=True
                    )
        except Exception as e:
            logger.error(f"{provider_name} API error: {e}")
            return _chat_response(f"Error calling {provider_name} API: {e}", error=True)

    main()
