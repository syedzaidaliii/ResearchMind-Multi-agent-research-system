"""Retrieval-augmented generation (RAG) helpers for ResearchMind.

Flow: search -> scrape pages in parallel -> chunk -> embed -> vector store ->
retrieve the best chunks -> hand them to the writer, which cites sources as [n].

Heavy libraries (Tavily, Gemini embeddings, Chroma, LangChain prompts) are
imported inside the functions that use them, so the rest of this module can be
tested without network access or API keys.
"""
import ipaddress
import os
import re
import socket
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urljoin, urlparse

import numpy as np
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=True)

# -- Settings (change here, or override the ones marked env in your .env file) --
MAX_RESULTS = int(os.getenv("RAG_MAX_RESULTS", "6"))   # sources fetched from Tavily
MAX_PAGE_CHARS = 30_000       # keep at most this much text per page
MAX_PAGE_BYTES = 1_500_000    # never download more than this per page
MIN_PAGE_CHARS = 300          # below this, fall back to the search snippet
CHUNK_SIZE = 1000             # characters per chunk
CHUNK_OVERLAP = 150           # characters repeated between neighbouring chunks
MAX_CHUNKS = 150              # cap on embedded chunks per run (free-tier friendly)
TOP_K = 10                    # chunks handed to the writer
FETCH_K = 30                  # candidates considered before diversifying
MMR_LAMBDA = 0.6              # 1.0 = pure relevance, 0.0 = pure diversity
EMBED_BATCH = 50              # texts per embedding request
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")  # env

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ResearchMind/2.0)"}


# =============================================================================
# 1. Search
# =============================================================================
def get_tavily():
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        raise RuntimeError("TAVILY_API_KEY is missing. Add it to your .env file.")
    from tavily import TavilyClient
    return TavilyClient(api_key=key)


def search_sources(query: str, max_results: int = MAX_RESULTS) -> list:
    """Return de-duplicated search results as [{n, title, url, snippet}]."""
    response = get_tavily().search(query=query, max_results=max_results,
                                   search_depth="basic")
    sources, seen = [], set()
    for r in response.get("results", []):
        url = (r.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        sources.append({
            "n": len(sources) + 1,
            "title": (r.get("title") or url).strip(),
            "url": url,
            "snippet": (r.get("content") or "").strip(),
        })
    return sources


# =============================================================================
# 2. Scrape
# =============================================================================
def is_public_url(url: str) -> bool:
    """Allow only http(s) URLs whose host resolves to public internet addresses."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    try:
        infos = socket.getaddrinfo(parsed.hostname, None)
    except OSError:
        return False
    for info in infos:
        try:
            if not ipaddress.ip_address(info[4][0]).is_global:
                return False
        except ValueError:
            return False
    return True


def extract_text(html) -> str:
    """Pull readable paragraph text out of an HTML page (str or bytes)."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header",
                     "aside", "form", "svg", "iframe"]):
        tag.decompose()
    root = soup.find("article") or soup.find("main") or soup.body or soup

    parts, seen = [], set()
    for el in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "blockquote"]):
        if el.name in ("li", "blockquote") and el.find(["p", "li"]):
            continue  # its inner paragraphs are collected on their own
        text = " ".join(el.get_text(" ", strip=True).split())
        is_heading = el.name.startswith("h")
        if not text or (len(text) < 30 and not is_heading) or text in seen:
            continue
        seen.add(text)
        parts.append(text)
    result = "\n".join(parts)

    if len(result) < 500:  # page built from bare <div>s: fall back to all text
        lines = [" ".join(ln.split()) for ln in root.get_text("\n", strip=True).splitlines()]
        result = "\n".join(ln for ln in lines if len(ln) >= 40)
    return result[:MAX_PAGE_CHARS]


def fetch_page_text(url: str, timeout=(5, 10)) -> str:
    """Download one page and return its text. Follows up to 3 redirects, checking each hop."""
    current = url
    for _ in range(4):
        if not is_public_url(current):
            raise ValueError("blocked: not a public web address")
        resp = requests.get(current, timeout=timeout, headers=HEADERS,
                            allow_redirects=False, stream=True)
        try:
            if resp.status_code in (301, 302, 303, 307, 308):
                location = resp.headers.get("Location")
                if not location:
                    raise ValueError("redirect without a location")
                current = urljoin(current, location)
                continue
            resp.raise_for_status()
            ctype = resp.headers.get("Content-Type", "").lower()
            if ctype and "html" not in ctype and "text" not in ctype:
                raise ValueError(f"unsupported content type: {ctype}")
            body = b""
            for part in resp.iter_content(65536):
                body += part
                if len(body) > MAX_PAGE_BYTES:
                    break
            return extract_text(body)
        finally:
            resp.close()
    raise ValueError("too many redirects")


def scrape_pages(sources: list, max_workers: int = 5) -> list:
    """Fetch all sources in parallel.

    Each result gets text plus a method: "scraped" (full page), "snippet" (the
    search snippet, used when the page could not be read) or "failed".
    """
    def work(src):
        try:
            text = fetch_page_text(src["url"])
        except Exception:
            text = ""
        if len(text) >= MIN_PAGE_CHARS:
            method = "scraped"
        elif len(src.get("snippet", "")) >= 40:
            text, method = src["snippet"], "snippet"
        else:
            text, method = "", "failed"
        return {**src, "text": text, "method": method, "chars": len(text)}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(work, sources))  # keeps the search order


# =============================================================================
# 3. Chunk
# =============================================================================
def _cut_point(text: str, size: int) -> int:
    """Best place to cut a long paragraph: a sentence end, else a space, else size."""
    window = text[:size]
    best = max(window.rfind(". "), window.rfind("? "), window.rfind("! "))
    if best >= size * 0.5:
        return best + 1
    space = window.rfind(" ")
    return space if space >= size * 0.5 else size


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    """Split text into chunks of about `size` characters on paragraph and sentence
    boundaries, repeating the last `overlap` characters at the start of the next chunk."""
    pieces = []
    for paragraph in (p.strip() for p in text.split("\n")):
        while len(paragraph) > size:
            cut = _cut_point(paragraph, size)
            pieces.append(paragraph[:cut].strip())
            paragraph = paragraph[cut:].strip()
        if paragraph:
            pieces.append(paragraph)

    chunks, current = [], ""
    for piece in pieces:
        if not current:
            current = piece
        elif len(current) + 1 + len(piece) <= size:
            current += "\n" + piece
        else:
            chunks.append(current)
            current = piece
    if current:
        chunks.append(current)

    if overlap > 0:
        overlapped = chunks[:1]
        for prev, chunk in zip(chunks, chunks[1:]):
            tail = prev[-overlap:]
            if " " in tail:
                tail = tail[tail.index(" ") + 1:]  # start on a whole word
            overlapped.append(f"{tail} {chunk}")
        chunks = overlapped

    return [c for c in chunks if len(c) >= 80] or chunks[:1]


def chunk_pages(pages: list) -> list:
    """Turn pages into [{text, n, url, title, chunk}] ready for embedding."""
    out = []
    for page in pages:
        for i, piece in enumerate(chunk_text(page["text"])):
            out.append({"text": piece, "n": page["n"], "url": page["url"],
                        "title": page["title"], "chunk": i})
    return out


def limit_chunks(chunks: list, max_total: int = MAX_CHUNKS) -> list:
    """Keep at most max_total chunks, taking turns between sources so no single page dominates."""
    if len(chunks) <= max_total:
        return chunks
    queues = {}
    for c in chunks:
        queues.setdefault(c["n"], []).append(c)
    ordered, depth = list(queues.values()), 0
    out = []
    while len(out) < max_total and any(depth < len(q) for q in ordered):
        for q in ordered:
            if depth < len(q) and len(out) < max_total:
                out.append(q[depth])
        depth += 1
    return out


# =============================================================================
# 4. Embed and store
# =============================================================================
def is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in ("429", "quota", "rate limit", "rate_limit",
                                  "resource_exhausted", "too many requests"))


class ResilientEmbeddings:
    """Wraps an embeddings model: sends texts in batches and retries on rate limits."""

    def __init__(self, base, batch_size: int = EMBED_BATCH, retries: int = 4,
                 sleep=time.sleep):
        self.base, self.batch_size, self.retries, self.sleep = base, batch_size, retries, sleep

    def _call(self, fn, *args):
        for attempt in range(self.retries + 1):
            try:
                return fn(*args)
            except Exception as exc:
                if not is_rate_limit_error(exc) or attempt == self.retries:
                    raise
                self.sleep(2 ** (attempt + 1))  # 2s, 4s, 8s, 16s

    def embed_documents(self, texts):
        vectors = []
        for i in range(0, len(texts), self.batch_size):
            vectors.extend(self._call(self.base.embed_documents,
                                      list(texts[i:i + self.batch_size])))
        return vectors

    def embed_query(self, text):
        return self._call(self.base.embed_query, text)


def get_embeddings():
    """Gemini embeddings (free tier) wrapped with batching and rate-limit retries."""
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    return ResilientEmbeddings(GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL))


class _Doc:
    """Minimal stand-in for a LangChain Document."""

    def __init__(self, page_content, metadata):
        self.page_content, self.metadata = page_content, metadata


class InMemoryVectorStore:
    """Small numpy vector store with cosine similarity and MMR search.

    Used automatically if Chroma is not installed or cannot start, and in tests.
    """

    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.texts, self.metadatas, self.matrix = [], [], None

    def add_texts(self, texts, metadatas=None):
        texts = list(texts)
        vectors = np.array(self.embeddings.embed_documents(texts), dtype=float)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = vectors / norms
        self.matrix = vectors if self.matrix is None else np.vstack([self.matrix, vectors])
        self.texts.extend(texts)
        self.metadatas.extend(metadatas or [{} for _ in texts])

    def max_marginal_relevance_search(self, query, k=4, fetch_k=20, lambda_mult=0.5):
        if self.matrix is None:
            return []
        q = np.array(self.embeddings.embed_query(query), dtype=float)
        q = q / (np.linalg.norm(q) or 1.0)
        sims = self.matrix @ q
        candidates = [int(i) for i in np.argsort(-sims)[:fetch_k]]
        selected = [candidates[0]]
        remaining = candidates[1:]
        while remaining and len(selected) < k:
            def score(i):
                redundancy = max(float(self.matrix[i] @ self.matrix[j]) for j in selected)
                return lambda_mult * float(sims[i]) - (1 - lambda_mult) * redundancy
            best = max(remaining, key=score)
            selected.append(best)
            remaining.remove(best)
        return [_Doc(self.texts[i], self.metadatas[i]) for i in selected]

    def delete_collection(self):
        self.texts, self.metadatas, self.matrix = [], [], None


def index_chunks(chunks: list, embeddings):
    """Embed the chunks and store them. Returns (store, backend_name).

    Each run uses its own fresh collection, so topics never mix. Chroma is used
    unless VECTOR_STORE=memory is set or Chroma cannot start.
    """
    texts = [c["text"] for c in chunks]
    metadatas = [{"n": c["n"], "url": c["url"], "title": c["title"], "chunk": c["chunk"]}
                 for c in chunks]

    if os.getenv("VECTOR_STORE", "chroma").lower() == "chroma":
        try:
            from langchain_chroma import Chroma
            store = Chroma(collection_name=f"run_{uuid.uuid4().hex[:12]}",
                           embedding_function=embeddings)
            store.add_texts(texts, metadatas=metadatas)
            return store, "chroma"
        except Exception as exc:
            if is_rate_limit_error(exc):
                raise
            print(f"[rag] Chroma unavailable ({type(exc).__name__}: {exc}). "
                  "Using the in-memory store instead.")

    store = InMemoryVectorStore(embeddings)
    store.add_texts(texts, metadatas=metadatas)
    return store, "memory"


def cleanup_store(store):
    try:
        store.delete_collection()
    except Exception:
        pass


# =============================================================================
# 5. Retrieve
# =============================================================================
def retrieve(store, query: str, k: int = TOP_K) -> list:
    """Return the k most relevant, non-repetitive chunks as [{n, url, title, chunk, text}]."""
    docs = store.max_marginal_relevance_search(
        query, k=k, fetch_k=max(FETCH_K, k), lambda_mult=MMR_LAMBDA)
    return [{"n": d.metadata["n"], "url": d.metadata["url"],
             "title": d.metadata["title"], "chunk": d.metadata.get("chunk", 0),
             "text": d.page_content} for d in docs]


def format_sources(pages: list) -> str:
    return "\n".join(f"[{p['n']}] {p['title']} ({p['url']})" for p in pages)


def format_context(chunks: list) -> str:
    """Group retrieved chunks by source so the writer sees one block per source."""
    groups = {}
    for c in chunks:
        groups.setdefault(c["n"], []).append(c)
    blocks = []
    for n in sorted(groups):
        items = sorted(groups[n], key=lambda c: c["chunk"])
        blocks.append(f"[{n}] {items[0]['title']}\n" + "\n---\n".join(c["text"] for c in items))
    return "\n\n=====\n\n".join(blocks)


# =============================================================================
# 6. Write with citations
# =============================================================================
WRITER_SYSTEM = (
    "You are a careful research writer. You write only from the numbered "
    "context excerpts you are given, and you cite every claim."
)

WRITER_HUMAN = """Write a detailed research report on the topic below using ONLY the context excerpts.

Topic: {topic}
Today's date: {today}

Sources:
{sources}

Context excerpts (each block starts with its source number):
{context}

Rules:
- Support every factual statement with a citation such as [1] or [2][3], using the source numbers above.
- Do not use outside knowledge. If the context does not cover something, say it could not be verified from the sources.
- If sources disagree, say so and cite each one.
- Never invent a source number.
- Do not write a Sources section. It is added automatically.

Structure the report as:
- Introduction
- Key Findings (at least 3 well-explained points)
- Conclusion

Be detailed, factual and professional."""


def build_writer_chain():
    """Prompt | Gemini | text, reusing the model defined in agents.py."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from agents import llm
    prompt = ChatPromptTemplate.from_messages([("system", WRITER_SYSTEM),
                                               ("human", WRITER_HUMAN)])
    return prompt | llm | StrOutputParser()


_SOURCES_HEADING = re.compile(r"\n\s*(?:#{1,6}\s*)?\**Sources\**\s*:?\s*\n", re.IGNORECASE)
_CITATION = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\]")


def strip_sources_section(report: str) -> str:
    """Remove a Sources section if the model wrote one anyway."""
    return _SOURCES_HEADING.split("\n" + report, maxsplit=1)[0].strip()


def extract_citations(text: str) -> set:
    """Source numbers cited in the text: [1], [1][2], [1, 3] and [2-4] are all understood."""
    found = set()
    for match in _CITATION.finditer(text):
        for part in re.split(r"\s*,\s*", match.group(1)):
            bounds = [int(x) for x in re.split(r"\s*[–-]\s*", part)]
            if len(bounds) == 2 and 0 <= bounds[1] - bounds[0] < 20:
                found.update(range(bounds[0], bounds[1] + 1))
            else:
                found.update(bounds)
    return {n for n in found if n < 1000}  # skip things like [2024]


def check_citations(report: str, valid_numbers) -> dict:
    """Compare the numbers the report cites with the sources that actually exist."""
    valid = set(valid_numbers)
    cited = extract_citations(strip_sources_section(report))
    return {
        "cited": sorted(cited & valid),
        "invalid": sorted(cited - valid),
        "uncited": sorted(valid - cited),
        "total": len(valid),
    }


def _clean_title(title: str) -> str:
    return " ".join(title.replace("[", "(").replace("]", ")").split())


def finalize_report(report: str, pages: list) -> str:
    """Append a Sources section listing the sources the report actually cites."""
    body = strip_sources_section(report)
    cited = extract_citations(body)
    listed = [p for p in pages if p["n"] in cited] or pages
    lines = [f"- **[{p['n']}]** [{_clean_title(p['title'])}]({p['url']})" for p in listed]
    return body + "\n\n## Sources\n\n" + "\n".join(lines)
