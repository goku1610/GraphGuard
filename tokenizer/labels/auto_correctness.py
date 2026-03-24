# --- tokenizer/labels/auto_correctness.py ---
import re

class AutoLabeler:
    """
    Evaluates the model's generated text against benchmark gold answers.
    Inspired by LLMsKnow compute_correctness.py
    """
    
    @staticmethod
    def extract_answer_tag(generated_text: str) -> str:
        """Surgically extracts the text inside the <answer> tags."""
        match = re.search(r"<answer>\n(.*?)\n</answer>", generated_text, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
        return "NO_ANSWER_TAG_FOUND"

    @staticmethod
    def evaluate_math(generated_answer: str, gold_answer: str) -> int:
        """
        Math Heuristic: Is the gold number anywhere in the final answer string?
        Returns 1 (Correct) or 0 (Incorrect).
        """
        # A simple string inclusion check (can be upgraded later)
        # If gold is '42', and answer is 'The answer is 42.', it returns 1.
        is_correct = (str(gold_answer).lower() in generated_answer.lower())
        return 1 if is_correct else 0

    @staticmethod
    def evaluate_nq(generated_answer: str, gold_answer: str) -> int:
        """
        Natural Questions Heuristic: String inclusion check.
        """
        # Similar logic: evaluate if the factual entity is present in the final output
        is_correct = (str(gold_answer).lower() in generated_answer.lower())
        return 1 if is_correct else 0

    def get_labels(self, generated_text: str, gold_answer: str, dataset_name: str) -> dict:
        """
        Takes the full generated text, extracts the final answer, and runs the correct heuristic.
        """
        extracted_answer = self.extract_answer_tag(generated_text)
        
        if "Math" in dataset_name:
            correctness = self.evaluate_math(extracted_answer, gold_answer)
        elif "Natural" in dataset_name:
            correctness = self.evaluate_nq(extracted_answer, gold_answer)
        else:
            # Fallback simple match
            correctness = 1 if gold_answer.lower() in extracted_answer.lower() else 0
            
        return {
            "correctness": correctness,
            "correctness_source": "benchmark_heuristic",
            "exact_answer_extracted": extracted_answer
        }