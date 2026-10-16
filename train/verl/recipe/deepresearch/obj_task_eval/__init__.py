"""
Mind2Web2 evaluation library package.
"""

__all__ = [
    "Evaluator",
    "AggregationStrategy",
    "LLMClient",
    "CacheFileSys",
    "VerificationNode",
    "create_evaluator",
    "Extractor",
    "Verifier",
    "BinaryEvalResult",
]


def __getattr__(name):
    if name in {"Evaluator", "AggregationStrategy"}:
        from .evaluator import Evaluator, AggregationStrategy

        return {"Evaluator": Evaluator, "AggregationStrategy": AggregationStrategy}[name]
    if name == "LLMClient":
        from .llm_client.base_client import LLMClient

        return LLMClient
    if name == "CacheFileSys":
        from .utils import CacheFileSys

        return CacheFileSys
    if name == "VerificationNode":
        from .verification_tree import VerificationNode

        return VerificationNode
    if name in {"create_evaluator", "Extractor", "Verifier", "BinaryEvalResult"}:
        from .eval_toolkit import BinaryEvalResult, Extractor, Verifier, create_evaluator

        return {
            "create_evaluator": create_evaluator,
            "Extractor": Extractor,
            "Verifier": Verifier,
            "BinaryEvalResult": BinaryEvalResult,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
