# Data Model: Web Research Tool

## Search Configuration

- `search_base_url`: HTTPS service base URL; default is Tavily's API endpoint.
- `search_api_key`: required secret; never serialized to traces or CLI output.
- `search_timeout`: positive time limit for one request.
- `search_max_results`: positive bounded number of source results.

## Web Search Arguments

- `query`: non-empty text, supplied by the model or the explicit-research
  preflight.
- `max_results`: optional positive override bounded by the configured maximum.

## Search Result

- `title`: source title, if available.
- `url`: absolute source URL.
- `content`: bounded source snippet.
- `score`: optional provider relevance score.

## Search Observation

- `query`: executed query.
- `results`: ordered normalized source results.
- `source_count`: number of usable sources.
- `provider_request_id`: optional non-secret provider correlation ID.
- `error_type`: absent on success; otherwise one of the documented search
  error categories.

## Trace Event

- `trace_type`: `tool_call`/`tool_result` through existing machinery, plus the
  existing task `turn` and state transition records.
- Required safe fields: task ID, action name, duration, result count, provider
  request ID, and error classification.
- Prohibited fields: API key, authorization header, full raw provider body when
  it could contain secrets.
