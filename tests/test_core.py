import unittest
from unittest.mock import patch

from src.data.chunking import split_by_article_or_window
from src.data.schema import LegalChunk, LegalDocument, SearchResult
from src.generation.gemini_client import BatchGeminiClient
from src.generation.prompt import build_context
from src.retrieval.pipeline import RetrievalPipeline


class DataPreparationTests(unittest.TestCase):
    def test_article_chunking_keeps_article_labels(self) -> None:
        text = "Điều 1. Nội dung thứ nhất. Điều 2. Nội dung thứ hai."
        self.assertEqual(
            split_by_article_or_window(text),
            [("Điều 1", "Điều 1. Nội dung thứ nhất."), ("Điều 2", "Điều 2. Nội dung thứ hai.")],
        )


class GeminiClientTests(unittest.TestCase):
    def test_default_rate_limit_stays_below_15_rpm_per_key(self) -> None:
        client = BatchGeminiClient(api_keys=["key-a", "key-b"])

        self.assertGreaterEqual(client.sleep_seconds, 5.0)
        self.assertEqual(set(client._last_request_at), {"key-a", "key-b"})


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

    def test_pipeline_returns_hybrid_results(self) -> None:
        pipeline = RetrievalPipeline(self.chunks, retrieval_method="hybrid", use_tfidf_fallback=True)
        results = pipeline.retrieve("trợ cấp thôi việc", top_k=2)

        self.assertEqual(pipeline.vector_backend, "tfidf")
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].chunk.chunk_id, "leave")
        self.assertEqual([result.rank for result in results], [1, 2])
        self.assertTrue(all(result.method == "hybrid" for result in results))

    def test_default_pipeline_uses_selected_method_with_rerank_fallback(self) -> None:
        pipeline = RetrievalPipeline(self.chunks, use_tfidf_fallback=True)

        with patch.object(pipeline, "_cross_encoder_rerank", side_effect=lambda _query, candidates: candidates):
            results = pipeline.retrieve("trợ cấp thôi việc", top_k=2)

        self.assertEqual(pipeline.retrieval_method, "hybrid_rrf_cross_encoder_mmr")
        self.assertEqual(len(results), 2)
        self.assertTrue(all(result.method == "hybrid_rrf_cross_encoder_mmr" for result in results))

    def test_prompt_context_contains_citation(self) -> None:
        result = SearchResult(self.chunks[0], 0.5, "hybrid", 1)
        context, citations = build_context([result])

        self.assertIn("[S1]", context)
        self.assertEqual(citations[0]["label"], "S1")
        self.assertEqual(citations[0]["chunk_id"], "leave")


if __name__ == "__main__":
    unittest.main()
