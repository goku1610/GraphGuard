# --- schema.py ---
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import torch

@dataclass
class UnifiedTraceRecord:
    """
    The Canonical Schema for a CHARM generation trace.
    Holds the text, the automated labels, and the internal signals.
    """
    # 1. Metadata & Text
    sample_id: str
    source_dataset: str
    prompt: str
    context: str  # The prompt context (useful for lookback)
    gold_answer: str
    generated_text: str
    exact_answer_extracted: str 
    
    # 2. Tiered Labels
    labels: Dict[str, int] = field(default_factory=dict)
    # Expected keys:
    # 'gold_correctness': 1 (correct) or 0 (incorrect)
    # 'human_hallucination': 1 (hallucinated) or 0 (faithful) - optional
    
    # 3. Dense & Sparse Signals
    activations: List[torch.Tensor] = field(default_factory=list)
    sparse_edges: List[list] = field(default_factory=list)
    lookback_ratios: List[float] = field(default_factory=list)
    
    # 4. Model & Generation Params
    metadata: Dict[str, Any] = field(default_factory=dict)