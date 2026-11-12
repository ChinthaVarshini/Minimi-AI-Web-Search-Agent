"""Tavily-backed web search used as the agent's only external tool."""

from urllib.parse import urlparse

import requests

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
MAX_RESULTS = 5
REQUEST_TIMEOUT = (3.05, 12)
MAX_QUERY_LENGTH = 400
MAX_CONTENT_LENGTH = 1_200


def search_web(query: str, api_key: str) -> list[dict[str, str]]:
    """Search the web and return citation-ready, bounded result snippets."""
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("The web search query cannot be empty.")
    if len(normalized_query) > MAX_QUERY_LENGTH:
        normalized_query = normalized_query[:MAX_QUERY_LENGTH]
    if not api_key:
        raise ValueError("Set TAVILY_API_KEY to enable web search.")

    response = requests.post(
        TAVILY_SEARCH_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "query": normalized_query,
            "search_depth": "basic",
            "max_results": MAX_RESULTS,
            "include_answer": False,
            "include_raw_content": False,
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    results = payload.get("results")
    if not isinstance(results, list):
        raise RuntimeError("The web search provider returned an invalid response.")

    citations = []
    for result in results:
        if not isinstance(result, dict):
            continue
        url = result.get("url")
        title = result.get("title")
        content = result.get("content")
        if not all(isinstance(value, str) for value in (url, title, content)):
            continue
        if urlparse(url).scheme not in {"http", "https"} or not urlparse(url).netloc:
            continue
        if not content.strip():
            continue
        citations.append(
            {
                "url": url,
                "title": title.strip() or url,
                "content": content.strip()[:MAX_CONTENT_LENGTH],
            }
        )
    return citations
