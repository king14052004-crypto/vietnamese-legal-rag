import os
import warnings

from src.evaluation.config import DEFAULT_EMBEDDING_MODEL, DEFAULT_MODEL
from src.evaluation.gemini_env import ensure_google_api_key


def gemini_model() -> str:
    return os.getenv("GEMINI_MODEL", DEFAULT_MODEL)


def build_langchain_llm(temperature: float = 0.0):
    api_key = ensure_google_api_key()
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=gemini_model(), google_api_key=api_key, temperature=temperature)


def build_langchain_embeddings():
    api_key = ensure_google_api_key()
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(model=DEFAULT_EMBEDDING_MODEL, google_api_key=api_key)


def build_ragas_llm():
    from ragas.llms.base import LangchainLLMWrapper

    return LangchainLLMWrapper(build_langchain_llm())


def build_ragas_embeddings():
    from ragas.embeddings.base import LangchainEmbeddingsWrapper

    return LangchainEmbeddingsWrapper(build_langchain_embeddings())


def retrieval_metrics():
    from ragas.metrics._context_precision import NonLLMContextPrecisionWithReference
    from ragas.metrics._context_recall import NonLLMContextRecall

    return [NonLLMContextPrecisionWithReference(), NonLLMContextRecall()]


def answer_metrics(include_answer_correctness: bool = True):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from ragas.metrics import answer_correctness, answer_relevancy, context_precision, context_recall, faithfulness

    answer_relevancy.strictness = 1
    metrics = [context_precision, context_recall, faithfulness, answer_relevancy]
    if include_answer_correctness:
        metrics.append(answer_correctness)
    return metrics
