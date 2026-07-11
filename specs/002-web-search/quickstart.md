# Quickstart: Web Research Tool

## Configure

Add the following values to `.agent_app/.env.local`; do not commit the key.

```ini
SEARCH_API_KEY=tvly-your-key
SEARCH_TIMEOUT=30
SEARCH_MAX_RESULTS=5
```

`SEARCH_BASE_URL` is optional and defaults to Tavily's public API endpoint.

## Validate a successful research request

```powershell
agent-app --workspace-root . "请查阅 Python 3.13 的发布信息，并给出来源链接"
```

Expected outcome: the final answer cites returned source URLs, and task traces
show a successful `web_search` result before the final model response.

## Validate unavailable configuration

Temporarily run without `SEARCH_API_KEY` and issue an explicit `查阅` request.

Expected outcome: a distinct search configuration failure, not `model_error`
and not an answer claiming that research was performed.

## Run regression tests

```powershell
python -m unittest discover -s tests
coverage run -m unittest discover -s tests
coverage report
```
