"""Offline tests: no API keys or network needed. Run: python -m unittest discover tests"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import rag


class FakeEmbeddings:
    """Deterministic bag-of-words vectors, so similar texts get similar vectors."""
    VOCAB = ["cat", "dog", "fish", "car", "engine", "wheel", "moon", "star"]

    def _vec(self, text):
        words = text.lower().split()
        v = [float(sum(w.startswith(k) for w in words)) for k in self.VOCAB]
        return v if any(v) else [0.01] * len(self.VOCAB)

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class ChunkTests(unittest.TestCase):
    def test_short_text_one_chunk(self):
        text = "A fairly normal sentence about research systems. " * 3
        self.assertEqual(len(rag.chunk_text(text)), 1)

    def test_long_text_splits_with_size_limit_and_overlap(self):
        text = "\n".join(f"Paragraph {i} " + "word " * 60 for i in range(30))
        chunks = rag.chunk_text(text, size=500, overlap=100)
        self.assertGreater(len(chunks), 3)
        for c in chunks:
            self.assertLessEqual(len(c), 500 + 100 + 5)
        # overlap: some words of chunk i appear at the start of chunk i+1
        tail_word = chunks[0].split()[-1]
        self.assertIn(tail_word, chunks[1][:150])

    def test_very_long_single_line(self):
        chunks = rag.chunk_text("Sentence number one is here. " * 200, size=400, overlap=50)
        self.assertTrue(all(len(c) <= 460 for c in chunks))

    def test_limit_chunks_round_robin(self):
        chunks = [{"n": 1, "text": f"a{i}"} for i in range(10)] + [{"n": 2, "text": "b0"}]
        out = rag.limit_chunks(chunks, 4)
        self.assertEqual(len(out), 4)
        self.assertIn(2, {c["n"] for c in out})


class ExtractTests(unittest.TestCase):
    def test_extract_text_drops_scripts_and_nav(self):
        html = ("<html><body><nav>Menu menu menu menu menu menu menu menu menu menu</nav>"
                "<script>var x=1;</script><article><h1>Title</h1>"
                "<p>" + "Real content sentence here. " * 5 + "</p></article></body></html>")
        text = rag.extract_text(html)
        self.assertIn("Real content", text)
        self.assertNotIn("var x", text)
        self.assertNotIn("Menu menu", text)

    def test_private_urls_blocked(self):
        for url in ("http://127.0.0.1/", "http://localhost:8000", "file:///etc/passwd",
                    "http://169.254.169.254/latest/meta-data", "ftp://example.com"):
            self.assertFalse(rag.is_public_url(url), url)


class StoreTests(unittest.TestCase):
    def test_mmr_returns_relevant_and_diverse(self):
        store = rag.InMemoryVectorStore(FakeEmbeddings())
        texts = ["cat cat cat", "cat cat cat", "cat cat dog", "car engine wheel", "moon star"]
        store.add_texts(texts, [{"n": i, "url": "u", "title": "t", "chunk": 0} for i in range(5)])
        top = store.max_marginal_relevance_search("cat", k=2, fetch_k=5, lambda_mult=0.4)
        self.assertEqual(len(top), 2)
        self.assertIn("cat", top[0].page_content)
        self.assertNotEqual(top[0].page_content, top[1].page_content)  # diversified

    def test_retrieve_shape(self):
        store = rag.InMemoryVectorStore(FakeEmbeddings())
        store.add_texts(["dog dog", "fish fish"],
                        [{"n": 1, "url": "u1", "title": "A", "chunk": 0},
                         {"n": 2, "url": "u2", "title": "B", "chunk": 0}])
        got = rag.retrieve(store, "dog", k=1)
        self.assertEqual(got[0]["n"], 1)
        self.assertEqual(got[0]["url"], "u1")

    def test_resilient_embeddings_batches_and_retries(self):
        class Flaky:
            calls = 0

            def embed_documents(self, texts):
                Flaky.calls += 1
                if Flaky.calls == 1:
                    raise RuntimeError("429 RESOURCE_EXHAUSTED")
                return [[1.0, 0.0] for _ in texts]

            def embed_query(self, t):
                return [1.0, 0.0]

        sleeps = []
        emb = rag.ResilientEmbeddings(Flaky(), batch_size=2, sleep=sleeps.append)
        vecs = emb.embed_documents(["a", "b", "c", "d", "e"])
        self.assertEqual(len(vecs), 5)
        self.assertEqual(sleeps, [2])

    def test_non_rate_limit_error_not_retried(self):
        class Bad:
            def embed_documents(self, t):
                raise ValueError("bad key")

        emb = rag.ResilientEmbeddings(Bad(), sleep=lambda s: None)
        with self.assertRaises(ValueError):
            emb.embed_documents(["a"])


class CitationTests(unittest.TestCase):
    def test_extract_forms(self):
        self.assertEqual(rag.extract_citations("a [1] b [2][3] c [4, 5] d [7-9] e [2024]"),
                         {1, 2, 3, 4, 5, 7, 8, 9})

    def test_check_citations(self):
        r = rag.check_citations("Fact [1]. Other [9].\n\n## Sources\n- [2] foo", [1, 2, 3])
        self.assertEqual(r["cited"], [1])
        self.assertEqual(r["invalid"], [9])
        self.assertEqual(r["uncited"], [2, 3])

    def test_finalize_lists_only_cited(self):
        pages = [{"n": i, "title": f"T{i} [x]", "url": f"http://e/{i}"} for i in (1, 2, 3)]
        out = rag.finalize_report("Body [1][3].\n\nSources:\n- junk", pages)
        self.assertIn("## Sources", out)
        self.assertNotIn("junk", out)
        self.assertIn("http://e/1", out)
        self.assertNotIn("http://e/2", out)
        self.assertIn("T1 (x)", out)

    def test_format_context_groups_by_source(self):
        ctx = rag.format_context([
            {"n": 2, "title": "B", "chunk": 0, "text": "bb"},
            {"n": 1, "title": "A", "chunk": 1, "text": "a2"},
            {"n": 1, "title": "A", "chunk": 0, "text": "a1"}])
        self.assertTrue(ctx.startswith("[1] A"))
        self.assertLess(ctx.index("a1"), ctx.index("a2"))
        self.assertLess(ctx.index("a2"), ctx.index("[2] B"))


if __name__ == "__main__":
    unittest.main()
