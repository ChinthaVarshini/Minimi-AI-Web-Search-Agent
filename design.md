# Architecture and Design

## Goals

Minimi is a small, responsive web-search assistant with readable citations and useful follow-up behavior. Its agent boundary is intentionally narrow: **web search is the only external tool made available to the model**. The chat model is the reasoning and response-generation service; it is not exposed as an agent tool.

## Components

```text
Browser
  |
  v
Streamlit UI (app.py)
  |  session history, timestamps, streamed rendering
  v
Agent orchestration (agent.py)
  |  OpenAI Chat Completions; only tool: web_search
  v
Web search adapter (search.py)
  |
  v
Tavily Search API
```

The OpenAI API handles model inference. Tavily is called only when the model invokes `web_search`; no browser automation, page fetching, file access, or other agent tools are registered.

## Request flow

1. The UI records the user's question with a local-timezone timestamp and adds it to Streamlit session history.
2. The agent sends a bounded slice of recent user/assistant messages to the chat model. This gives the model context for follow-ups such as "What about the second option?"
3. For factual questions, the model can call `web_search` with a focused query informed by that history.
4. `search.py` calls Tavily with a bounded result count and timeout. It accepts only HTTP(S) result URLs and returns titles, URLs, and short snippets rather than full pages.
5. The agent assigns stable citation numbers across search calls, returns the results to the model, and continues the tool cycle within a small limit.
6. The final answer streams into the chat. Citation numbers are checked against the results returned for that turn, and the UI presents those results as clickable source links.
7. The assistant answer, its timestamp, and its sources are added to session history for display and follow-up context.

## Key decisions

### One explicit tool

The model receives a single function schema for `web_search`. The application rejects unknown function names and limits tool rounds. The search provider adapter is the only tool implementation; adding page browsing, code execution, or other capabilities requires an explicit design change.

### Search snippets rather than page scraping

Tavily provides result snippets and canonical URLs. Limiting the amount of external text reduces latency and prompt size, and avoids adding a second page-fetching capability. The trade-off is that answers are limited by the quality and coverage of search snippets.

### Citation numbering

Each distinct result URL receives a stable number for the current response. The model is instructed to cite factual claims with those numbers. The UI renders the result titles and URLs so readers can verify the underlying source. Invalid citation numbers are removed; when the model omits citations, the response includes the retrieved references.

### Session-scoped memory

Recent message content is kept in `st.session_state` and sent with follow-up requests. The model context is bounded to the most recent messages and each message is length-limited. This keeps the demo simple and avoids storing conversations on a server; memory is not durable across sessions or guaranteed across browser refreshes.

### Streaming and latency

The app streams model output, uses a lightweight default model, asks Tavily for at most five results, bounds search requests with a timeout, and limits tool rounds. These choices favor responsive interaction over exhaustive research.

### Credentials and untrusted content

API keys are read from environment variables or entered into password fields. `.env` is excluded from version control. Search snippets are treated as untrusted evidence, not instructions; the system prompt explicitly warns the model not to follow instructions in retrieved content.

## Limitations

- Search snippets may be incomplete, stale, or inconsistent. Citations support verification but do not guarantee correctness.
- Search results are not full-page retrieval; important details absent from snippets may be missed.
- The default conversation memory is session-scoped and in-memory, not persistent storage.
- Provider rate limits, API availability, network conditions, and model tool-calling behavior affect latency and reliability.
- The tool-round limit can stop a complex research question before it is fully answered.
