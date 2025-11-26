from macrec.tools.base import Tool
from macrec.tools.summarize import TextSummarizer
from macrec.tools.wikipedia import Wikipedia
from macrec.tools.info_database import InfoDatabase
from macrec.tools.interaction import InteractionRetriever
from macrec.tools.rag_sql import RAGTextToSQL
from macrec.tools.collaborative_filtering import CollaborativeFiltering
from macrec.tools.rating_predictor import RatingPredictor
from macrec.tools.sequential_predictor import SequentialPredictor

TOOL_MAP: dict[str, type] = {
    'summarize': TextSummarizer,
    'wikipedia': Wikipedia,
    'info': InfoDatabase,
    'interaction': InteractionRetriever,
    'rag_sql': RAGTextToSQL,
    'collaborative_filtering': CollaborativeFiltering,
    'rating_predictor': RatingPredictor,
    'sequential_predictor': SequentialPredictor,
}
