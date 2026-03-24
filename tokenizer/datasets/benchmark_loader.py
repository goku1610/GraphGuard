# --- tokenizer/datasets/benchmark_loader.py ---
import os
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class BenchmarkExample:
    """A single benchmark question mapped to a standard format before generation."""
    sample_id: str
    dataset_name: str
    split: str
    prompt: str
    gold_answer: str
    context: Optional[str] = None # Used for Lookback tasks like Summarization/RAG

class BenchmarkLoader:
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            tokenizer_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(tokenizer_root, "data", "llmsknow")
        self.data_dir = data_dir

    def load_math(self, split="test") -> List[BenchmarkExample]:
        """Loads AnswerableMath.csv and maps it to BenchmarkExample."""
        filename = "AnswerableMath_test.csv" if split == "test" else "AnswerableMath.csv"
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"⚠️ Warning: Dataset not found at {filepath}")
            return []
            
        df = pd.read_csv(filepath)
        examples = []
        for idx, row in df.iterrows():
            # LLMsKnow Math gold answers are lists like "['42']", we extract the first element safely
            raw_gold = row['answer']
            gold_str = eval(raw_gold)[0] if isinstance(raw_gold, str) and raw_gold.startswith('[') else str(raw_gold)
            
            ex = BenchmarkExample(
                sample_id=f"math_{split}_{idx}",
                dataset_name="AnswerableMath",
                split=split,
                prompt=row['question'],
                gold_answer=str(gold_str)
            )
            examples.append(ex)
        return examples

    def load_nq(self, split="test") -> List[BenchmarkExample]:
        """Loads Natural Questions dataset."""
        candidate_filenames = [f"nq_wc_dataset_{split}.csv", "nq_wc_dataset.csv"]
        filepath = None
        for filename in candidate_filenames:
            candidate_path = os.path.join(self.data_dir, filename)
            if os.path.exists(candidate_path):
                filepath = candidate_path
                break
        
        if filepath is None:
            print(f"⚠️ Warning: Dataset not found in {self.data_dir} for split '{split}'")
            return []
            
        df = pd.read_csv(filepath)
        examples = []
        for idx, row in df.iterrows():
            ex = BenchmarkExample(
                sample_id=f"nq_{split}_{idx}",
                dataset_name="NaturalQuestions",
                split=split,
                prompt=row['Question'],
                gold_answer=str(row['Answer']),
                context=row.get('Context', None) # NQ has optional context
            )
            examples.append(ex)
        return examples