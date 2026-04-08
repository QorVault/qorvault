"""BoardDocs RAG Search Tool for Open WebUI
=========================================

Allows a conversational LLM to search the Kent School District BoardDocs
document corpus for factual information about board meetings, policies,
budgets, and district operations.

External service dependencies:
  - BoardDocs RAG API: HTTP on port 8000 (default)
    Provides semantic search over ~20,000 BoardDocs documents with
    Claude-generated answers and source citations.
    From inside the Open WebUI container, reach it via:
      http://host.containers.internal:8000

Installation:
  Open WebUI → Workspace → Tools → "+" → paste this entire file → Save

Configuration (via Valves in the Open WebUI tool settings):
  - RAG_API_URL: Base URL of the RAG API (default uses host.containers.internal
    which resolves to the host machine from inside any Podman container on Linux)
  - TENANT_ID: District identifier (default: kent_sd)
  - TOP_K: Number of document chunks to retrieve per query (default: 10)
  - TIMEOUT_SECONDS: HTTP request timeout (default: 60, needs to be high because
    the RAG API calls Claude which can take 10-20 seconds)
"""

from pydantic import BaseModel, Field


class Tools:
    class Valves(BaseModel):
        RAG_API_URL: str = Field(
            default="http://host.containers.internal:8000",
            description="Base URL of the BoardDocs RAG API (use host.containers.internal from inside Podman containers)",
        )
        TENANT_ID: str = Field(
            default="kent_sd",
            description="Tenant ID for BoardDocs queries (for future multi-district support)",
        )
        TOP_K: int = Field(
            default=10,
            description="Number of document chunks to retrieve per query",
        )
        TIMEOUT_SECONDS: int = Field(
            default=60,
            description="HTTP request timeout in seconds (RAG pipeline calls Claude, which can take 10-20s)",
        )

    def __init__(self):
        self.valves = self.Valves()

    async def search_boarddocs(
        self,
        query: str,
        __event_emitter__=None,
    ) -> str:
        """Search the Kent School District BoardDocs corpus for information.
        Use this tool whenever the user asks about Kent School District board meetings,
        board decisions, school district policies, district budgets, personnel matters,
        superintendent reports, committee discussions, public hearings, resolutions,
        or any other official Kent School District business.
        Do NOT use this tool for general knowledge questions unrelated to Kent School District.

        :param query: The search query describing what information is needed from BoardDocs
        :return: Relevant information from BoardDocs documents with source citations
        """
        import aiohttp

        url = f"{self.valves.RAG_API_URL}/api/v1/query"
        payload = {
            "query": query,
            "top_k": self.valves.TOP_K,
        }

        # Emit status so the user sees a loading indicator
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Searching BoardDocs...",
                        "done": False,
                    },
                }
            )

        try:
            timeout = aiohttp.ClientTimeout(total=self.valves.TIMEOUT_SECONDS)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        return (
                            f"Error: The BoardDocs search API returned status {resp.status}. "
                            f"The service may be temporarily unavailable. Details: {error_text[:200]}"
                        )
                    data = await resp.json()
        except aiohttp.ClientConnectorError:
            return (
                f"Error: Could not connect to the BoardDocs RAG API at {url}. "
                f"The service may not be running. Please try again later."
            )
        except TimeoutError:
            return (
                f"Error: The BoardDocs search timed out after {self.valves.TIMEOUT_SECONDS} seconds. "
                f"Please try a more specific query."
            )
        except aiohttp.ClientError as exc:
            return f"Error: HTTP error querying BoardDocs: {exc}"
        except Exception as exc:
            return f"Error: Unexpected error querying BoardDocs: {type(exc).__name__}: {exc}"

        # Parse the RAG API response
        answer = data.get("answer", "No answer was returned from the search.")
        citations = data.get("citations", [])
        chunks_retrieved = data.get("chunks_retrieved", 0)

        # Emit completion status
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Found {chunks_retrieved} relevant document chunks",
                        "done": True,
                    },
                }
            )

        # Format citations as a clean markdown block
        if not citations:
            return answer

        parts = [answer, "", "---", "**Sources:**", ""]
        for citation in citations:
            title = citation.get("title") or "Untitled"
            meta = []
            if citation.get("committee_name"):
                meta.append(citation["committee_name"])
            if citation.get("meeting_date"):
                meta.append(citation["meeting_date"])
            meta_str = f" ({', '.join(meta)})" if meta else ""
            url_str = f" — {citation['source_url']}" if citation.get("source_url") else ""
            parts.append(f"- [Source {citation.get('source_number', '?')}] {title}{meta_str}{url_str}")

        return "\n".join(parts)
