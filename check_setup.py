"""Quick check that your keys, models and vector store work. Run: python check_setup.py"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=True)


def check(name, fn):
    try:
        detail = fn()
        print(f"[ OK ] {name}" + (f": {detail}" if detail else ""))
        return True
    except Exception as exc:
        print(f"[FAIL] {name}: {type(exc).__name__}: {str(exc)[:300]}")
        return False


def keys():
    missing = [k for k in ("GOOGLE_API_KEY", "TAVILY_API_KEY") if not os.getenv(k)]
    if missing:
        raise RuntimeError("missing in .env: " + ", ".join(missing))


def tavily():
    import rag
    return f"{len(rag.search_sources('python programming', 2))} results"


def llm():
    from agents import llm as model
    return str(model.invoke("Reply with the single word: ok").content)[:40]


def embeddings():
    import rag
    vec = rag.get_embeddings().embed_query("hello world")
    return f"model {rag.EMBEDDING_MODEL}, {len(vec)} dimensions"


def chroma():
    import rag
    store, backend = rag.index_chunks(
        [{"text": "Paris is the capital of France.", "n": 1, "url": "x", "title": "t", "chunk": 0}],
        rag.get_embeddings())
    rag.cleanup_store(store)
    return f"vector store backend: {backend}"


results = [check("API keys in .env", keys)]
if results[0]:
    results += [check("Tavily search", tavily), check("Gemini chat model", llm),
                check("Gemini embeddings", embeddings), check("Vector store", chroma)]
sys.exit(0 if all(results) else 1)
