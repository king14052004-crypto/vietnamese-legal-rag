import unittest
from unittest.mock import patch

from src.data.chunking import split_by_article_or_window
from src.data.filter_labor import is_labor_related
from src.data.schema import LegalChunk, LegalDocument, SearchResult
from src.generation.prompt import build_context
from src.retrieval.pipeline import RetrievalPipeline


class DataPreparationTests(unittest.TestCase):
    def test_labor_filter_accepts_strong_phrase(self) -> None:
        document = LegalDocument(
            id="labor",
            content_text="Quy định về hợp đồng lao động.",
        )
        self.assertTrue(is_labor_related(document))

    def test_labor_filter_rejects_unrelated_document(self) -> None:
        document = LegalDocument(id="other", title="Quy hoạch sử dụng đất")
        self.assertFalse(is_labor_related(document))

    def test_article_chunking_keeps_article_labels(self) -> None:
        text = "Điều 1. Nội dung thứ nhất. Điều 2. Nội dung thứ hai."
        self.assertEqual(
            split_by_article_or_window(text),
            [("Điều 1", "Điều 1. Nội dung thứ nhất."), ("Điều 2", "Điều 2. Nội dung thứ hai.")],
        )


class RetrievalPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = [
            LegalChunk(
                "leave",
                "doc-1",
                "Trợ cấp thôi việc",
                "Người lao động được nhận trợ cấp thôi việc.",
            ),
            LegalChunk("salary", "doc-2", "Tiền lương", "Quy định về lương tối thiểu vùng."),
            LegalChunk(
                "safety",
                "doc-3",
                "An toàn lao động",
                "Trách nhiệm bảo đảm an toàn vệ sinh lao động.",
            ),
        ]

    def test_pipeline_rejects_empty_corpus(self) -> None:
        with self.assertRaises(ValueError):
            RetrievalPipeline([], use_tfidf_fallback=True)

    def test_pipeline_returns_hybrid_rrf_results(self) -> None:
        pipeline = RetrievalPipeline(self.chunks, use_tfidf_fallback=True)
        results = pipeline.retrieve("trợ cấp thôi việc", top_k=2)

        self.assertEqual(pipeline.vector_backend, "tfidf")
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].chunk.chunk_id, "leave")
        self.assertEqual([result.rank for result in results], [1, 2])
        self.assertTrue(all(result.method == "hybrid_rrf" for result in results))

    def test_large_tfidf_fallback_uses_keyword_scan(self) -> None:
        chunks = [
            LegalChunk("match", "doc-1", "Labor contract", "employee termination notice period"),
            LegalChunk("other", "doc-2", "Salary", "minimum wage region"),
            LegalChunk("safety", "doc-3", "Safety", "workplace safety equipment"),
        ]
        with patch("src.retrieval.pipeline.LARGE_CORPUS_CHUNK_THRESHOLD", 2):
            pipeline = RetrievalPipeline(chunks, use_tfidf_fallback=True)

        results = pipeline.retrieve("employee termination notice", top_k=1)

        self.assertEqual(pipeline.vector_backend, "keyword_scan")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk.chunk_id, "match")
        self.assertEqual(results[0].method, "hybrid_rrf")

    def test_prompt_context_contains_citation(self) -> None:
        result = SearchResult(self.chunks[0], 0.5, "hybrid_rrf", 1)
        context, citations = build_context([result])

        self.assertIn("[S1]", context)
        self.assertEqual(citations[0]["label"], "S1")
        self.assertEqual(citations[0]["chunk_id"], "leave")


if __name__ == "__main__":
    unittest.main()
