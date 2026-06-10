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
