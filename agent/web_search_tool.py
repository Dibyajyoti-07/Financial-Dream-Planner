import os

from langchain_core.tools import tool
from tavily import TavilyClient

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    return _client


@tool
def web_search(query: str) -> list[dict]:
    """Search the live web for current, real-world information (e.g. investment products, market options, general financial news). Returns a list of {title, url, content} results. Use only for qualitative research - never as a source for a financial figure or calculation."""
    result = _get_client().search(query, max_results=5)
    return [{"title": r["title"], "url": r["url"], "content": r["content"]} for r in result["results"]]
