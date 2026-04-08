#!/usr/bin/env python3
"""Test script for the BoardDocs RAG Open WebUI tool.

Instantiates the tool class directly and calls search_boarddocs()
to verify the HTTP logic works before installing in Open WebUI.

Requires the RAG API to be running on 127.0.0.1:8000.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path so we can import the tool
sys.path.insert(0, str(Path(__file__).parent))

from boarddocs_rag_tool import Tools


async def main():
    tool = Tools()
    # Override URL for host-side testing (inside the container it uses
    # host.containers.internal, but from the host we use 127.0.0.1)
    tool.valves.RAG_API_URL = "http://127.0.0.1:8000"

    query = "school board budget discussion"
    print(f"Query: {query!r}")
    print("=" * 70)

    result = await tool.search_boarddocs(query)
    print(result)

    print("=" * 70)
    print(f"Result length: {len(result)} chars")

    if result.startswith("Error:"):
        print("\nTEST FAILED — tool returned an error")
        print("Is the RAG API running on 127.0.0.1:8000?")
        sys.exit(1)
    elif "[Source" in result:
        print("\nTEST PASSED — answer with citations returned")
    else:
        print("\nTEST WARNING — got an answer but no citations")


if __name__ == "__main__":
    asyncio.run(main())
