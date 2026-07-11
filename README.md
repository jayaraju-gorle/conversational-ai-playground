# Conversational AI Playground

A self-hosted playground that helps businesses evaluate **AI-driven sales and support agents** before committing to a stack. Try a hospital appointment-booking bot, a bank support agent, or an e-commerce order-support assistant — over **live voice** or **text chat** — while switching every layer of the stack and watching real cost, latency, and token metrics.

Built on [Pipecat](https://github.com/pipecat-ai/pipecat) (cascade pipeline: STT → LLM → TTS) with a single-file web UI.

## Features

- **Scenario templates** — General Assistant, Hospital (appointment booking), Bank (customer support), E-commerce (order support). Each swaps the agent's persona and greeting.
- **Swappable providers** — mix and match at session start:
  - **LLM**: Gemini 2.5 Flash, Sarvam 30B, GPT-4.1, Llama 3.3 70B (Groq)
  - **STT**: Sarvam Saaras v3, Deepgram Nova
  - **TTS**: Sarvam Bulbul v3 (25 voices), Cartesia Sonic
- **10 Indian languages** — English, Hindi, Bengali, Gujarati, Kannada, Malayalam, Marathi, Punjabi, Tamil, Telugu.
- **Voice and text modes** — real-time WebRTC voice sessions, or direct LLM text chat.
- **Real metrics, not estimates** — cost (with STT/LLM/TTS breakup), voice-to-voice latency (with per-service TTFB), and token usage come from the pipeline's own metrics and provider usage reports. Per-message metrics on every reply, cumulative metrics per session.
- **USD / INR** currency toggle (configurable FX rate).
- **Compare mode** — run two agents side by side in one page with independent configs. Speak once and both agents hear you, or type once and it goes to both. Compare cost/latency/quality directly.

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

   You need at least one key per layer. The playground adapts to what's configured:

   | Layer | Providers (env var) |
   |---|---|
   | LLM | `GEMINI_API_KEY`, `SARVAM_API_KEY`, `OPENAI_API_KEY`, `GROQ_API_KEY` |
   | STT | `SARVAM_API_KEY`, `DEEPGRAM_API_KEY` |
   | TTS | `SARVAM_API_KEY`, `CARTESIA_API_KEY` |

4. **Run the playground**:

   ```bash
   uv run bot.py
   ```

   Open http://localhost:7860

## Pricing data

Cost figures are computed from **real usage** (LLM tokens, TTS characters, audio minutes) at the rates in the `PRICING` table at the top of `server/bot.py`. The shipped rates are estimates — verify them against each vendor's pricing page and adjust before relying on them. The USD→INR rate is `USD_INR_RATE` in `.env`.

## Project Structure

```
conversational-ai-playground/
├── server/                    # Python bot server
│   ├── bot.py                 # Pipeline, provider catalog, scenarios, pricing, APIs
│   ├── templates/client.html  # Playground web UI (single file)
│   ├── pyproject.toml         # Python dependencies
│   ├── .env.example           # Environment variables template
│   ├── Dockerfile             # Container image for Pipecat Cloud
│   └── pcc-deploy.toml        # Pipecat Cloud deployment config
├── .gitignore
└── README.md
```

## HTTP API

| Endpoint | Purpose |
|---|---|
| `GET /` | Playground UI |
| `GET /api/config` | Provider/voice/language/scenario catalog + pricing |
| `POST /api/chat` | Direct text chat (returns content + real token usage + latency) |
| `POST /start` | Start a voice session (Pipecat runner; body carries the selected config) |

## Production deployment (GCE VM)

The playground is served from a GCE VM — **https://35-234-215-193.sslip.io** — because
WebRTC voice media (UDP) cannot leave Cloud Run, which this project originally deployed to.

| Item | Value |
|---|---|
| GCP project | `gen-lang-client-0981591737` |
| VM | `playground-vm` (e2-small, zone `asia-south1-a`) |
| Static IP | `playground-ip` = `35.234.215.193` (regional, `asia-south1`) |
| URL | `https://35-234-215-193.sslip.io` (Caddy auto-HTTPS via Let's Encrypt; sslip.io resolves the hostname to the IP) |
| App stack | `/opt/app/docker-compose.yml` on the VM: `app` (image from Artifact Registry) + `caddy` (TLS on 80/443) |
| Env vars | `/opt/app/.env` on the VM |
| Firewall | `playground-web` (tcp 80/443, target tag `playground`) |

**CI/CD**: pushing to `main` runs `.github/workflows/deploy.yml`, which builds the image
with Cloud Build and pushes `:latest` + `:<sha>` to Artifact Registry
(`asia-south1-docker.pkg.dev/gen-lang-client-0981591737/cloud-run-apps/conversational-ai-playground`).
On the VM, a systemd timer (`deploy.timer` → `/opt/app/deploy.sh`) polls the registry every
2 minutes and redeploys when `:latest` changes (log: `/var/log/playground-deploy.log`).

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

This project is also configured for deployment to Pipecat Cloud. See the [Pipecat Cloud Documentation](https://docs.pipecat.ai/deployment/pipecat-cloud/introduction) to learn about configuring, deploying, and managing agents.

## Building with an AI coding agent

Extending this project with Claude Code, Codex, or another AI coding assistant? Give it live, accurate Pipecat context instead of stale training data with the **Pipecat Context Hub** — a local index of Pipecat docs, examples, and API source your agent queries over MCP:

```bash
claude mcp add pipecat-context-hub -- uvx pipecat-ai-context-hub serve
```

See `AGENTS.md` for agent-oriented guidance on working with Pipecat.
