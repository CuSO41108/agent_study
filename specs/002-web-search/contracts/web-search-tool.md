# `web_search` Tool Contract

## Availability

The root coordinator exposes `web_search` alongside local tools when a search
provider is configured. A request that explicitly requires research runs one
preflight search; the model may make additional calls only within ordinary task
budgets.

## Function Schema

```json
{
  "type": "function",
  "function": {
    "name": "web_search",
    "description": "Search the public web and return source-backed results.",
    "parameters": {
      "type": "object",
      "additionalProperties": false,
      "required": ["query"],
      "properties": {
        "query": {"type": "string", "minLength": 1},
        "max_results": {"type": "integer", "minimum": 1}
      }
    }
  }
}
```

## Successful Result

The tool result content is a bounded JSON-safe textual summary containing the
query and one or more source objects with `title`, `url`, and `content`.

## Failure Result

The result is unsuccessful and identifies one of:

- `search_configuration_error`
- `search_request_error`
- `search_http_error`
- `search_invalid_response`
- `search_no_results`

No result may include an authentication credential.
