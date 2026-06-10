# Claire-s

## AIHubMix configuration

The LLM layer uses AIHubMix's OpenAI-compatible API:

- Base URL: `https://aihubmix.com/v1`
- Chat endpoint: `/chat/completions`
- Default model: `gpt-4o-mini`

Set your key in the runtime environment before starting the app or running a
manual smoke test:

```bash
export AIHUBMIX_API_KEY="sk-..."
```

Optional overrides:

```bash
export AIHUBMIX_MODEL="gpt-4o-mini"
export AIHUBMIX_BASE_URL="https://aihubmix.com/v1"
export AIHUBMIX_TIMEOUT_SECONDS="60"
```

Manual smoke test:

```bash
python api.py "Say hello in one sentence"
```

## Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Run the API server

Start the FastAPI app:

```bash
uvicorn api:app --reload --port 8000
```

If your shell cannot find `uvicorn`, run it through Python:

```bash
python -m uvicorn api:app --reload --port 8000
```

Then open:

```text
http://localhost:8000/docs
```

Useful demo endpoints:

- `GET /context` - returns delayed public market context from Stooq
- `POST /chat` - sends a simple prompt to AIHubMix
- `POST /interpret-indicator` - generates macro analysis for an indicator
