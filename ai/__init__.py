"""
Package AI — agents LLM, pipeline Text2SQL complet.
"""

from ai.intent_agent      import IntentAgent, OllamaClient, RuleBasedFallback, create_intent_agent
from ai.table_matcher     import TableMatcher, HybridScorer, create_table_matcher
from ai.sql_generator     import SQLGenerator
from ai.sql_validator     import SQLValidator
from ai.missing_attributes import MissingAttributesDetector
from ai.pipeline          import AIPipeline, PipelineResult, create_pipeline
from ai.prompts           import (
    SYSTEM_INTENT_ANALYZER,
    SYSTEM_TABLE_PREDICTOR,
    SYSTEM_ATTRIBUTE_EXTRACTOR,
    SYSTEM_ACTION_CLASSIFIER,
)

__all__ = [
    "IntentAgent", "OllamaClient", "RuleBasedFallback", "create_intent_agent",
    "TableMatcher", "HybridScorer", "create_table_matcher",
    "SQLGenerator",
    "SQLValidator",
    "MissingAttributesDetector",
    "AIPipeline", "PipelineResult", "create_pipeline",
    "SYSTEM_INTENT_ANALYZER", "SYSTEM_TABLE_PREDICTOR",
    "SYSTEM_ATTRIBUTE_EXTRACTOR", "SYSTEM_ACTION_CLASSIFIER",
]
