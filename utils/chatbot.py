import os
import json
import requests

# ── Groq API (Free: 14,400 req/day, 30 req/min) ───────────────────────────────
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.3-70b-versatile"

KEY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'api_key.txt')

_data_context       = {}
_saved_api_key      = ""
_cached_system_prompt = ""


# ── Key persistence ───────────────────────────────────────────────────────────
def _load_key_from_file():
    global _saved_api_key
    try:
        if os.path.exists(KEY_FILE):
            with open(KEY_FILE, 'r') as f:
                k = f.read().strip()
                if k:
                    _saved_api_key = k
    except Exception:
        pass

_load_key_from_file()


def save_api_key(key: str):
    global _saved_api_key
    _saved_api_key = key.strip()
    try:
        os.makedirs(os.path.dirname(KEY_FILE), exist_ok=True)
        with open(KEY_FILE, 'w') as f:
            f.write(_saved_api_key)
    except Exception as e:
        print(f"[chatbot] Could not save key: {e}")


def get_saved_key() -> str:
    return _saved_api_key


# ── Data context ──────────────────────────────────────────────────────────────
def set_data_context(context: dict):
    """Called once at boot — caches full system prompt so every chat is 1 API call."""
    global _data_context, _cached_system_prompt
    _data_context = context
    _cached_system_prompt = _build_system_prompt(context)
    print("[chatbot] System prompt cached — ready for chat.")


def _build_system_prompt(ctx: dict) -> str:
    forecast_total = sum(
        f.get('Predicted_Revenue', 0) for f in ctx.get('revenue_forecast', [])
    )
    fm = ctx.get('forecast_metrics', {})

    return f"""You are RetailIQ — an expert AI business analyst for an online retail company.
You have COMPLETE access to the dataset below. Answer every question using the ACTUAL numbers
from this data. Never say data is unavailable if it is listed below. Be specific and concise.

════════════════════════════════════════
FULL DATASET CONTEXT
════════════════════════════════════════
{ctx.get('summary', 'No summary loaded.')}

════════════════════════════════════════
ML MODEL — REVENUE FORECAST
════════════════════════════════════════
Algorithm : Gradient Boosting Regressor
R² Score  : {fm.get('r2', 'N/A')}
MAE       : £{fm.get('mae', 'N/A')}
RMSE      : £{fm.get('rmse', 'N/A')}
MAPE      : {fm.get('mape', 'N/A')}%

════════════════════════════════════════
CUSTOMER SEGMENTS (K-Means + RFM)
════════════════════════════════════════
{json.dumps(ctx.get('segment_counts', []), indent=2)}

════════════════════════════════════════
TOP 10 PRODUCTS BY REVENUE
════════════════════════════════════════
{json.dumps(ctx.get('top_products', [])[:10], indent=2)}

════════════════════════════════════════
30-DAY REVENUE FORECAST
════════════════════════════════════════
Predicted 30-day total : £{forecast_total:,.0f}
First 7 days breakdown :
{json.dumps(ctx.get('revenue_forecast', [])[:7], indent=2)}

════════════════════════════════════════
RESPONSE RULES
════════════════════════════════════════
- Always use exact numbers from the data above
- Always display currency as £ (e.g. £447,137) — NEVER write GBP
- NEVER say data is unavailable if it appears above
- Keep answers under 300 words unless deep analysis is requested
- Use markdown bold and bullet points for clarity
- End every answer with one clear actionable recommendation
- Clearly distinguish forecast (predicted) from historical (actual) data
"""


def _resolve_key(api_key: str) -> str:
    if api_key and len(api_key) > 10:
        return api_key.strip()
    if _saved_api_key and len(_saved_api_key) > 10:
        return _saved_api_key
    return os.environ.get("GROQ_API_KEY", "").strip()


# ── Chat — exactly ONE Groq API call per message ──────────────────────────────
def chat(message: str, history: list, api_key: str) -> str:
    key = _resolve_key(api_key)

    if not key:
        return (
            "⚠️ **No Groq API key found.**\n\n"
            "**Steps to fix:**\n"
            "1. Go to **https://console.groq.com/keys**\n"
            "2. Sign up free — no credit card needed\n"
            "3. Click **Create API Key**, copy it (starts with `gsk_`)\n"
            "4. Paste it in the **🔑 API** box in the top navbar\n"
            "5. Click **Save**\n\n"
            "✅ Free tier: **14,400 requests/day · 30/minute**"
        )

    # Use pre-built system prompt (built once at boot)
    system_prompt = _cached_system_prompt or _build_system_prompt(_data_context)

    # Last 8 turns only — keeps token count low and responses fast
    messages = [{"role": "system", "content": system_prompt}]
    for turn in history[-8:]:
        messages.append({"role": "user",      "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})
    messages.append({"role": "user", "content": message})

    payload = {
        "model":       GROQ_MODEL,
        "messages":    messages,
        "max_tokens":  800,
        "temperature": 0.4,
        "stream":      False,
    }

    try:
        response = requests.post(
            GROQ_API_URL,
            headers={
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {key}",
            },
            json=payload,
            timeout=30,
        )

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]

        elif response.status_code == 400:
            err = response.json().get("error", {}).get("message", "Bad request")
            return f"❌ **Bad request:** {err}"

        elif response.status_code == 401:
            return (
                "❌ **Invalid Groq API key.**\n\n"
                "- Make sure key is copied from https://console.groq.com/keys\n"
                "- Key must start with `gsk_`\n"
                "- No extra spaces"
            )

        elif response.status_code == 413:
            return "⚠️ **Context too long.** Click **Clear** to reset the chat and try again."

        elif response.status_code == 429:
            try:
                err_msg     = response.json().get('error', {}).get('message', '')
                retry_after = response.headers.get('retry-after', '60')
            except Exception:
                err_msg, retry_after = '', '60'

            if 'day' in err_msg.lower():
                return (
                    "⏳ **Daily limit reached** (14,400 req/day free tier).\n\n"
                    "Resets at midnight UTC. Get a new free key at https://console.groq.com/keys"
                )
            return (
                f"⏳ **Rate limit hit.** Wait ~{retry_after}s then try again.\n\n"
                "✅ Your key is working fine — just 30 req/min on free tier."
            )

        elif response.status_code == 503:
            return "🔄 **Groq is temporarily busy.** Please try again in a moment."

        else:
            return f"❌ **API error {response.status_code}:** {response.text[:300]}"

    except requests.exceptions.Timeout:
        return "⏳ **Request timed out.** Please try again."
    except requests.exceptions.ConnectionError:
        return "🔌 **Connection error.** Check your internet and try again."
    except Exception as e:
        return f"❌ **Unexpected error:** {str(e)}"


# ── One-click insights ────────────────────────────────────────────────────────
def get_quick_insights(api_key: str, ctx: dict) -> str:
    if not _resolve_key(api_key):
        return "⚠️ Please save your Groq API key first (`gsk_...`) — get one free at https://console.groq.com/keys"

    if ctx and not _cached_system_prompt:
        set_data_context(ctx)

    prompt = (
        "Using the retail dataset in your context, provide exactly **5 key strategic insights**.\n\n"
        "Format each as:\n"
        "**1. [Short Title]**\n"
        "[2-3 sentences with specific numbers from the data]\n"
        "💡 Action: [one concrete recommendation]\n\n"
        "Cover: revenue trends, best/worst months, top products, customer segments, and geographic opportunities."
    )
    return chat(prompt, [], api_key)
