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

    def load_movie_qa(self, split="test") -> List[BenchmarkExample]:
        filename = "movie_qa_test.csv" if split == "test" else "movie_qa_train.csv"
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            print(f"⚠️ Warning: Dataset not found at {filepath}")
            return []

        df = pd.read_csv(filepath)
        examples = []
        for idx, row in df.iterrows():
            examples.append(
                BenchmarkExample(
                    sample_id=f"movieqa_{split}_{idx}",
                    dataset_name="MovieQA",
                    split=split,
                    prompt=str(row["Question"]),
                    gold_answer=str(row["Answer"]),
                )
            )
        return examples

    def load_mnli(self, split="train") -> List[BenchmarkExample]:
        filename = "mnli_validation.csv" if split == "validation" else "mnli_train.csv"
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            print(f"⚠️ Warning: Dataset not found at {filepath}")
            return []

        df = pd.read_csv(filepath)
        examples = []
        for idx, row in df.iterrows():
            examples.append(
                BenchmarkExample(
                    sample_id=f"mnli_{split}_{idx}",
                    dataset_name="MNLI",
                    split=split,
                    prompt=str(row["Question"]),
                    gold_answer=str(row["Answer"]),
                )
            )
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

    def load_winogrande(self, split="test") -> List[BenchmarkExample]:
        filename = "winogrande_test.csv" if split == "test" else "winogrande_train.csv"
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            print(f"⚠️ Warning: Dataset not found at {filepath}")
            return []

        df = pd.read_csv(filepath)
        examples = []
        for idx, row in df.iterrows():
            examples.append(
                BenchmarkExample(
                    sample_id=f"winogrande_{split}_{idx}",
                    dataset_name="WinoGrande",
                    split=split,
                    prompt=str(row["Question"]),
                    gold_answer=str(row["Answer"]),
                )
            )
        return examples

    def load_winobias(self, split="test") -> List[BenchmarkExample]:
        filename = "winobias_test.csv" if split == "test" else "winobias_dev.csv"
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            print(f"⚠️ Warning: Dataset not found at {filepath}")
            return []

        df = pd.read_csv(filepath)
        examples = []
        for idx, row in df.iterrows():
            prompt = row.get("q_instruct", row.get("q", ""))
            context = row.get("sentence", None)
            examples.append(
                BenchmarkExample(
                    sample_id=f"winobias_{split}_{idx}",
                    dataset_name="WinoBias",
                    split=split,
                    prompt=str(prompt),
                    gold_answer=str(row["answer"]),
                    context=str(context) if context is not None else None,
                )
            )
        return examples

    def load_dataset(self, dataset: str, split: str = "test") -> List[BenchmarkExample]:
        key = dataset.lower()
        if key == "math":
            return self.load_math(split=split)
        if key == "nq":
            return self.load_nq(split=split)
        if key == "movie_qa":
            return self.load_movie_qa(split=split)
        if key == "mnli":
            normalized_split = "validation" if split in {"val", "valid", "validation", "test"} else "train"
            return self.load_mnli(split=normalized_split)
        if key == "winogrande":
            return self.load_winogrande(split=split)
        if key == "winobias":
            return self.load_winobias(split=split)
        print(f"⚠️ Warning: Unsupported dataset key '{dataset}'.")
        return []