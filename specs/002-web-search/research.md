# Research: Web Research Tool

## Decision: Tavily Search API over direct scraping or a new SDK

Use Tavily's `POST https://api.tavily.com/search` endpoint through the standard
library HTTP client already used by the model adapter. Send a Bearer API key and
a bounded JSON request with the user query, `search_depth: "basic"`, no
provider-generated answer, and a configured `max_results` value.

## Rationale

- The API returns ranked result objects with title, URL and content snippets,
  which map directly to source-backed Agent observations.
- The endpoint exposes standard HTTP status behavior, so configuration, timeout,
  HTTP and JSON/schema failures can be classified without a new dependency.
- `basic` search has a documented one-credit cost and bounded response shape;
  the model—not the provider—will synthesize the answer from supplied sources.

## Alternatives Considered

- **Direct search-engine scraping**: rejected because unstable page formats,
  consent pages and terms create poor reliability and observability.
- **Tavily SDK**: rejected because the project deliberately depends on the
  standard library plus `jsonschema`; the HTTP contract is small.
- **Provider-generated answer**: rejected because the tool should provide
  traceable sources to the ReAct loop rather than conceal synthesis inside the
  search provider.

## Decision: Explicit research intent is a runtime precondition

An LLM normally chooses tools autonomously through ReAct. That is appropriate
for ordinary requests, but an explicit user requirement to research must not be
silently ignored when the model returns a plausible answer from memory. Detect a
small, documented set of explicit research phrases (including `查阅`, `联网检索`,
`搜索网页`, `search the web`, and `look up`) at task start and run one bounded,
read-only `web_search` preflight. Supply the resulting observation to the
existing model-driven ReAct loop.

## Rationale

This preserves ReAct's autonomy for planning, follow-up search calls, file
creation, verification, and all other tools while guaranteeing the user-visible
contract that "查阅" actually retrieves sources. It is a policy guard, not a
new planner/supervisor or multi-agent layer.

## Error Semantics

`search_configuration_error`, `search_request_error`, `search_http_error`,
`search_invalid_response`, and `search_no_results` are terminal, recognizable
outcomes for required research. They must not be rewritten as `model_error`.
