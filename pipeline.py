from datetime import date

from agents import build_reader_agent, build_search_agent, writer_chain, critic_chain

# (key, title, description) -- the UI uses this to draw the step tracker
STEPS = [
    ("search", "Search Agent", "Finds recent sources on the web"),
    ("reader", "Reader Agent", "Scrapes the most relevant page"),
    ("writer", "Writer", "Drafts the research report"),
    ("critic", "Critic", "Reviews and scores the report"),
]

# which key of the final `state` dict each step fills
STATE_KEYS = {
    "search": "search_results",
    "reader": "scraped_content",
    "writer": "report",
    "critic": "feedback",
}


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


def _used_tool(result) -> bool:
    """True if the agent actually called a tool (its run contains a tool message)."""
    return any(getattr(m, "type", "") == "tool" for m in result["messages"])


STRICT_SEARCH = (
    "You MUST call the web_search tool before answering and use ONLY what it "
    "returns. Never answer from memory. List every source with its title and URL."
)


def run_research_steps(topic: str):
    """Run the pipeline step by step.

    Yields dicts so a UI can show live progress:
        {"step": "search", "status": "running"}
        {"step": "search", "status": "done", "key": "search_results", "output": "..."}
    """
    state = {}

    # step 1 - search agent
    yield {"step": "search", "status": "running"}
    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]
    })
    if not _used_tool(search_result):
        # the model answered from memory instead of searching -- try once more, strictly
        search_result = search_agent.invoke({
            "messages": [("user", f"{STRICT_SEARCH}\n\nTopic: {topic}")]
        })
        if not _used_tool(search_result):
            raise RuntimeError(
                "The search agent did not run a web search, so the report would not be "
                "based on real sources. Please try again."
            )
    state["search_results"] = _text(search_result["messages"][-1].content)
    yield {"step": "search", "status": "done", "key": "search_results",
           "output": state["search_results"]}

    # step 2 - reader agent
    yield {"step": "reader", "status": "running"}
    reader_agent = build_reader_agent()
    reader_result = reader_agent.invoke({
        "messages": [("user",
            f"Based on the following search results about '{topic}', "
            f"pick the most relevant URL and scrape it for deeper content.\n\n"
            f"Search Results:\n{state['search_results'][:3000]}"
        )]
    })
    state["scraped_content"] = _text(reader_result["messages"][-1].content)
    yield {"step": "reader", "status": "done", "key": "scraped_content",
           "output": state["scraped_content"]}

    # step 3 - writer chain
    yield {"step": "writer", "status": "running"}
    research_combined = (
        f"TODAY'S DATE: {date.today():%d %B %Y}\n\n"
        f"SEARCH RESULTS : \n {state['search_results']} \n\n"
        f"DETAILED SCRAPED CONTENT : \n {state['scraped_content']}"
    )
    state["report"] = _text(writer_chain.invoke({
        "topic": topic,
        "research": research_combined,
    }))
    yield {"step": "writer", "status": "done", "key": "report",
           "output": state["report"]}

    # step 4 - critic chain
    yield {"step": "critic", "status": "running"}
    state["feedback"] = _text(critic_chain.invoke({"report": state["report"]}))
    yield {"step": "critic", "status": "done", "key": "feedback",
           "output": state["feedback"]}


def run_research_pipeline(topic: str) -> dict:
    """Terminal version: runs every step, prints the output, returns the state."""
    titles = {key: title for key, title, _ in STEPS}
    state = {}

    for event in run_research_steps(topic):
        if event["status"] == "running":
            print("\n" + "=" * 50)
            print(f"{titles[event['step']]} is working ...")
            print("=" * 50)
        else:
            state[event["key"]] = event["output"]
            print(f"\n{event['key']}:\n", event["output"])

    return state


if __name__ == "__main__":
    topic = input("\n Enter a research topic : ")
    run_research_pipeline(topic)