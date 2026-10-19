from datetime import date

import rag
from agents import critic_chain

# (key, title, description) -- the UI uses this to draw the step tracker
STEPS = [
    ("search", "Search", "Finds sources on the web (Tavily)"),
    ("scrape", "Read pages", "Downloads several pages in parallel"),
    ("index", "Index", "Chunks the text, embeds it, stores it in a vector database"),
    ("retrieve", "Retrieve", "Picks the most relevant, non-repetitive chunks"),
    ("writer", "Writer", "Writes the report and cites sources as [n]"),
    ("critic", "Critic", "Reviews and scores the report"),
]


def _text(content) -> str:
    """Return plain text from a model message's content.

    Gemini models can return a list of content blocks instead of a string,
    so normalise both shapes to a single str.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type", "text") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


def run_research_steps(topic: str):
    """Run the pipeline step by step.

    Yields dicts so a UI can show live progress:
        {"step": "search", "status": "running"}
        {"step": "search", "status": "done", "key": "sources", "output": [...]}
        {"status": "data", "key": "citations", "output": {...}}   (extra results, no step)
    """
    store = None
    try:
        # 1 - search
        yield {"step": "search", "status": "running"}
        sources = rag.search_sources(topic)
        if not sources:
            raise RuntimeError("The web search returned no results. Try rewording the topic.")
        yield {"step": "search", "status": "done", "key": "sources", "output": sources}

        # 2 - scrape pages in parallel
        yield {"step": "scrape", "status": "running"}
        pages = [p for p in rag.scrape_pages(sources) if p["text"]]
        if not pages:
            raise RuntimeError("None of the pages could be read. Try a different topic.")
        yield {"step": "scrape", "status": "done", "key": "pages", "output": pages}

        # 3 - chunk, embed, store
        yield {"step": "index", "status": "running"}
        chunks = rag.limit_chunks(rag.chunk_pages(pages))
        store, backend = rag.index_chunks(chunks, rag.get_embeddings())
        yield {"step": "index", "status": "done", "key": "index_info",
               "output": {"chunks": len(chunks), "pages": len(pages), "backend": backend}}

        # 4 - retrieve
        yield {"step": "retrieve", "status": "running"}
        top = rag.retrieve(store, topic)
        yield {"step": "retrieve", "status": "done", "key": "retrieved", "output": top}

        # 5 - write with citations
        yield {"step": "writer", "status": "running"}
        writer = rag.build_writer_chain()
        raw = _text(writer.invoke({
            "topic": topic,
            "today": f"{date.today():%d %B %Y}",
            "sources": rag.format_sources(pages),
            "context": rag.format_context(top),
        }))
        report = rag.finalize_report(raw, pages)
        yield {"step": "writer", "status": "done", "key": "report", "output": report}
        yield {"status": "data", "key": "citations",
               "output": rag.check_citations(raw, [p["n"] for p in pages])}

        # 6 - critic
        yield {"step": "critic", "status": "running"}
        feedback = _text(critic_chain.invoke({"report": report}))
        yield {"step": "critic", "status": "done", "key": "feedback", "output": feedback}
    finally:
        if store is not None:
            rag.cleanup_store(store)


def run_research_pipeline(topic: str) -> dict:
    """Terminal version: runs every step, prints a short summary, returns the results."""
    titles = {key: title for key, title, _ in STEPS}
    state = {}

    for event in run_research_steps(topic):
        if event["status"] == "running":
            print(f"\n[{titles[event['step']]}] working ...")
            continue
        state[event["key"]] = event["output"]
        if event["key"] == "report":
            print("\n" + event["output"])
        elif event["key"] == "feedback":
            print("\n" + event["output"])
        elif event["key"] == "citations":
            c = event["output"]
            print(f"\nCitations: {len(c['cited'])}/{c['total']} sources cited, "
                  f"{len(c['invalid'])} invalid")
    return state


if __name__ == "__main__":
    topic = input("\n Enter a research topic : ")
    run_research_pipeline(topic)
