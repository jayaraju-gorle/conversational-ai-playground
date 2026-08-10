# Conversational AI Playground

A self-hosted playground that helps businesses evaluate **AI-driven sales and support agents** before committing to a stack. Try a hospital appointment-booking bot, a bank support agent, an e-commerce order assistant, a restaurant table reservation agent, a hotel room booking assistant, or a general assistant — over **live voice** (WebRTC cascade or native Speech-to-Speech) or **direct text chat** — while switching every layer of the stack and observing real cost, latency, and token metrics.

Built on [Pipecat](https://github.com/pipecat-ai/pipecat) (cascade pipeline: STT → LLM → TTS, or native Speech-to-Speech) with a single-file web UI.

## Features

- **6 Scenario Templates** — General Assistant, Hospital (appointment booking), Bank (customer support), E-commerce (order support), Restaurant (table reservation), Hotel (room booking). Each configures the agent's persona, system instructions, and sample user utterances.
- **Swappable Providers & Stack Layers** — mix and match at session start:
  - **LLM**:
    - *Direct APIs*: Gemini 3.5 Flash, Gemini 3.5 Flash-Lite, GPT-4.1, GPT-5.4, Sarvam 30B, Sarvam 105B, Claude Opus 4.8, Claude Sonnet 5, Claude Haiku 4.5
    - *Hosted on Groq*: Llama 3.3 70B, Llama 3.1 8B Instant, OpenAI GPT-OSS 120B, OpenAI GPT-OSS 20B, Groq Compound, Groq Compound Mini, Llama 4 Scout 17B (preview), Qwen3 32B (preview), Qwen3.6 27B (preview)
    - *Hosted on Azure*: OpenAI GPT-5 mini
    - *Hosted on GCP Vertex AI*: Gemini 3.5 Flash, Gemini 2.5 Pro, xAI Grok 4.1 Fast, xAI Grok 4.3
  - **Realtime (Speech-to-Speech / S2S)**: Gemini Live (S2S), OpenAI Realtime (S2S), OpenAI Realtime mini (S2S). Replaces the cascade with native voice-to-voice models and native voice selection (e.g. Puck, Charon, Kore, Fenrir, Aoede, Leda, Orus, Zephyr, marin, cedar, alloy, ash, ballad, coral, echo, sage, shimmer, verse).
  - **STT**: Sarvam Saaras v3, Deepgram Nova, Whisper v3 (Groq), Whisper v3 Turbo (Groq), GPT-4o Transcribe (Azure)
  - **TTS**: Sarvam Bulbul v3 (25 voices), Cartesia Sonic, Deepgram Aura-2 (9 English voices), Groq Orpheus (Canopy Labs, 6 English voices)
- **30 Multilingual Languages** — 12 Indic languages (*English India, Hindi, Bengali, Gujarati, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu, Urdu*) + 18 International languages (*English US, English UK, Spanish, French, German, Italian, Portuguese, Dutch, Polish, Russian, Turkish, Arabic, Mandarin, Japanese, Korean, Indonesian, Vietnamese, Thai*). The UI automatically filters supported languages per provider combination.
- **Voice and Text Modes** — real-time WebRTC voice sessions, or direct LLM text chat with exact token usage and response latency.
- **Real Metrics, Not Estimates** — cost breakup per layer (STT per minute, LLM per 1M prompt/completion tokens, TTS per 1k characters, or S2S audio tokens), voice-to-voice latency (with per-service TTFB), and token counts directly from pipeline metrics and provider response headers.
- **Live USD / INR Currency Toggle** — real-time exchange rates auto-fetched from ECB reference rates (`frankfurter.app`, cached 6h) with configurable offline fallback (`USD_INR_RATE`).
- **Compare Mode** — run two agents side by side in one page with independent configs. Speak once and both agents hear you simultaneously, or type once to send to both. Directly compare latency, cost, and response quality.
- **Headless Evaluation Mode** — run scripted benchmark scenarios without a live browser connection using `uv run bot.py -t eval` powered by Pipecat's `EvalTransportParams`.
- **Model Catalog Auto-Sync** — `GET /api/models/sync` diffs the curated catalog against live provider model endpoints to highlight newly launched models or deprecated candidates across Groq, Anthropic, Gemini, OpenAI, and Azure.
- **WebRTC & TURN Relay Support** — fully configurable ICE/TURN server integration (`TURN_URLS`, `TURN_SERVER_SIDE`, `TURN_SERVER_URLS`) for WebRTC sessions behind strict corporate firewalls or Cloud Run NAT environments.

## Setup

### Server

1. **Navigate to server directory**:

   ```bash
   cd server
   ```

2. **Install dependencies**:

   ```bash
   uv sync
   ```

3. **Configure environment variables**:

   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

   The playground dynamically enables providers based on configured environment variables:

   | Layer | Supported Env Vars |
   |---|---|
   | LLM (Direct) | `GEMINI_API_KEY` / `GOOGLE_API_KEY`, `OPENAI_API_KEY`, `SARVAM_API_KEY`, `ANTHROPIC_API_KEY` |
   | LLM (Groq) | `GROQ_API_KEY` |
   | LLM (Azure) | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` |
   | LLM (Vertex) | `GOOGLE_APPLICATION_CREDENTIALS`, `GOOGLE_VERTEX_PROJECT_ID` |
   | Realtime (S2S) | `GEMINI_API_KEY` / `GOOGLE_API_KEY`, `OPENAI_API_KEY` |
   | STT | `SARVAM_API_KEY`, `DEEPGRAM_API_KEY`, `GROQ_API_KEY`, `AZURE_OPENAI_API_KEY` |
   | TTS | `SARVAM_API_KEY`, `CARTESIA_API_KEY`, `DEEPGRAM_API_KEY`, `GROQ_API_KEY` |
   | WebRTC Relay | `TURN_URLS`, `TURN_USERNAME`, `TURN_PASSWORD`, `TURN_SERVER_SIDE`, `TURN_SERVER_URLS` |

4. **Run the playground**:

   ```bash
   uv run bot.py
   ```

   Open http://localhost:7860

5. **Headless Evaluation**:

   To run scripted scenarios headlessly without the browser UI:

   ```bash
   uv run bot.py -t eval
   ```

## Pricing Data & FX Rates

Cost figures are computed from **real usage** (LLM tokens, TTS characters, audio minutes, S2S audio tokens) using the `PRICING` table in `server/bot.py`. Currency conversion uses live exchange rates from `frankfurter.app` (ECB reference rates), falling back to `USD_INR_RATE` in `.env` if offline. Shipped rates are estimates — verify them against each vendor's pricing page.

## Project Structure

```
conversational-ai-playground/
├── server/                    # Python bot server & pipeline
│   ├── bot.py                 # Pipeline, catalog, pricing, scenarios, APIs, & WebSocket/WebRTC runners
│   ├── templates/client.html  # Playground web UI (single-file HTML/JS/CSS)
│   ├── pyproject.toml         # Dependencies managed by uv
│   ├── .env.example           # Environment variable template
│   ├── Dockerfile             # Container image for Pipecat Cloud / GCE deployment
│   ├── Dockerfile.cloudrun    # Cloud Run build image
│   └── pcc-deploy.toml        # Pipecat Cloud deployment config
├── .github/
│   └── workflows/
│       └── deploy.yml         # CI/CD workflow for GCP Artifact Registry & GCE deployment
├── cloudbuild.yaml            # Cloud Build configuration
├── AGENTS.md                  # Development & architectural context for AI agents
└── README.md
```

## HTTP API

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | `GET` | Playground Web UI |
| `/api/config` | `GET` | Dynamic provider catalog, voices, languages, scenarios, pricing, live FX rates, and ICE/TURN servers |
| `/api/chat` | `POST` | Direct text chat endpoint (returns assistant message, token usage, latency, and model name) |
| `/api/models/sync` | `GET` | Diffs hand-curated provider catalog against live provider model endpoints (Groq, Anthropic, Gemini, OpenAI, Azure) |
| `/start` | `POST` | Start a WebRTC voice session (Pipecat runner with selected pipeline configuration) |

## Production Deployment (GCE VM)

The playground is served from a GCE VM — **https://35-234-215-193.sslip.io** — because WebRTC voice media (UDP) requires direct network transport that Cloud Run standard HTTP routing does not support without TURN relays.

| Item | Value |
|---|---|
| GCP Project | `gen-lang-client-0981591737` |
| VM | `playground-vm` (e2-small, zone `asia-south1-a`) |
| Static IP | `playground-ip` = `35.234.215.193` (regional, `asia-south1`) |
| URL | `https://35-234-215-193.sslip.io` (Caddy auto-HTTPS via Let's Encrypt; sslip.io resolves hostname to IP) |
| App Stack | `/opt/app/docker-compose.yml` on VM: `app` (Artifact Registry image) + `caddy` (TLS on 80/443) |
| Env Vars | `/opt/app/.env` on VM |
| Firewall | `playground-web` (tcp 80/443, target tag `playground`) |

**CI/CD**: Pushing to `main` triggers `.github/workflows/deploy.yml`, which builds the container image via Cloud Build and pushes `:latest` + `:<sha>` to Google Artifact Registry (`asia-south1-docker.pkg.dev/gen-lang-client-0981591737/cloud-run-apps/conversational-ai-playground`). On the VM, a systemd timer (`deploy.timer` → `/opt/app/deploy.sh`) polls the registry every 2 minutes and automatically redeploys when `:latest` changes (log: `/var/log/playground-deploy.log`).

Useful commands:

```bash
# SSH into the VM
gcloud compute ssh playground-vm --project gen-lang-client-0981591737 --zone asia-south1-a

# App logs / status / manual redeploy (on the VM)
sudo docker logs -f app-app-1
sudo docker compose -f /opt/app/docker-compose.yml ps
sudo /opt/app/deploy.sh
```

## Deploying to Pipecat Cloud

This project is also configured for deployment to Pipecat Cloud using `server/pcc-deploy.toml` and `server/Dockerfile`. Refer to the [Pipecat Cloud Documentation](https://docs.pipecat.ai/deployment/pipecat-cloud/introduction) for setup details.

## Building with an AI Coding Agent

Extending this project with an AI coding assistant? Give it live, accurate Pipecat context with the **Pipecat Context Hub** — a local index of Pipecat docs, examples, and API source that your agent queries over MCP:

```bash
claude mcp add pipecat-context-hub -- uvx pipecat-ai-context-hub serve
```

See `AGENTS.md` for agent-oriented architectural guidance on working with Pipecat in this repository.

