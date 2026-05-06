"""Search client — backed by Wikipedia REST API (no key required)."""

from __future__ import annotations

import json
import logging
import re
import urllib.parse
import urllib.request

import wikipediaapi

from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)

_SNIPPET_CHARS = 300
_WIKI_USER_AGENT = "multi-agent-research-lab/0.1 (educational project)"

# Words to strip when building a keyword-only fallback query
_STOPWORDS = {
    "what", "is", "are", "how", "does", "do", "why", "when", "where", "who",
    "which", "a", "an", "the", "and", "or", "in", "on", "at", "to", "of",
    "for", "with", "it", "its", "explain", "describe", "tell", "me", "about",
}


def _extract_keywords(query: str) -> str:
    """Return significant words from query (strip question words)."""
    words = re.findall(r"[A-Za-z0-9]+", query)
    keywords = [w for w in words if w.lower() not in _STOPWORDS]
    return " ".join(keywords) if keywords else query


class SearchClient:
    """Wikipedia-backed search client.  No API key required."""

    def __init__(self) -> None:
        self._wiki = wikipediaapi.Wikipedia(
            language="en",
            user_agent=_WIKI_USER_AGENT,
        )

    def _opensearch(self, query: str, limit: int) -> tuple[list[str], list[str]]:
        """Call Wikipedia opensearch and return (titles, urls)."""
        encoded = urllib.parse.quote(query)
        url = (
            f"https://en.wikipedia.org/w/api.php"
            f"?action=opensearch&search={encoded}&limit={limit}&format=json"
        )
        req = urllib.request.Request(url, headers={"User-Agent": _WIKI_USER_AGENT})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        titles: list[str] = data[1] if len(data) > 1 else []
        urls: list[str] = data[3] if len(data) > 3 else []
        return titles, urls

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search Wikipedia for pages relevant to *query*.

        Uses opensearch; if zero results, retries with keyword-only query.
        """
        logger.info("SearchClient.search | query=%r max_results=%d", query, max_results)

        try:
            titles, urls = self._opensearch(query, max_results)
            if not titles:
                # Retry with keywords only
                kw_query = _extract_keywords(query)
                logger.info("SearchClient: retry with keywords=%r", kw_query)
                titles, urls = self._opensearch(kw_query, max_results)
        except Exception as exc:
            logger.warning("Wikipedia opensearch failed: %s", exc)
            return []

        documents: list[SourceDocument] = []
        for title, page_url in zip(titles[:max_results], urls[:max_results]):
            page = self._wiki.page(title)
            if not page.exists():
                continue
            snippet = page.summary[:_SNIPPET_CHARS].strip()
            if not snippet:
                continue
            documents.append(
                SourceDocument(
                    title=title,
                    url=page_url,
                    snippet=snippet,
                    metadata={"full_url": page.fullurl},
                )
            )

        logger.info("SearchClient.search | found %d documents", len(documents))
        return documents
