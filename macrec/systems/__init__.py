from macrec.systems.base import System
from macrec.systems.react import ReActSystem
from macrec.systems.reflection import ReflectionSystem
from macrec.systems.chat import ChatSystem
from macrec.systems.analyse import AnalyseSystem
from macrec.systems.collaboration import CollaborationSystem

# Import other systems if they exist
try:
    from macrec.systems.itemknn import ItemKNNSystem
except ImportError:
    ItemKNNSystem = None

try:
    from macrec.systems.hybrid import HybridSystem
except ImportError:
    HybridSystem = None

# Create SYSTEMS list with CollaborationSystem first (as default)
SYSTEMS: list[type[System]] = [CollaborationSystem]  # Default first
SYSTEMS.extend([value for value in [ReActSystem, ReflectionSystem, ChatSystem, AnalyseSystem] + ([ItemKNNSystem] if ItemKNNSystem else []) + ([HybridSystem] if HybridSystem else []) if value and value != CollaborationSystem])
