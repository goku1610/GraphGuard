# --- tokenizer/collectors/collect_benchmark.py ---
import os
import sys
import argparse
import json
import torch
from tqdm import tqdm
from transformers import StoppingCriteria, StoppingCriteriaList

# Ensure Python can find our modular imports if run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grammar import build_generator
from extractors.trace_extractor import TraceExtractor
from datasets.benchmark_loader import BenchmarkLoader, BenchmarkExample
from labels.auto_correctness import AutoLabeler
from storage.schema import UnifiedTraceRecord
from storage.sample_writer import TraceDatasetManager
from live_viewer.event_bus import LiveDemoReporter, LiveEventBus

# --- Utilities (Ported cleanly from old main.py) ---
class StopOnTag(StoppingCriteria):
    def __init__(self, tokenizer, stop_tag="</confidence>", reporter=None, max_generated_tokens=1024):
        self.tokenizer = tokenizer
        self.stop_tag = stop_tag
        self.reporter = reporter
        self.prompt_token_count = 0
        self.max_generated_tokens = max_generated_tokens
        self.stop_reason = None
        tokenizer_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.control_path = os.path.join(tokenizer_root, "live_viewer", "runtime", "control.json")

    def configure_sample(self, prompt_token_count: int):
        self.prompt_token_count = prompt_token_count
        self.stop_reason = None

    def __call__(self, input_ids, scores, **kwargs):
        tail_tokens = input_ids[0][-15:]
        tail_text = self.tokenizer.decode(tail_tokens)
        generated_ids = input_ids[0][self.prompt_token_count:]

        if self._consume_skip_signal():
            self.stop_reason = "manual_skip"
            return True

        if len(generated_ids) >= self.max_generated_tokens:
            self.stop_reason = "max_generated_tokens"
            return True

        if self.reporter is not None:
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
            self.reporter.update_generated_text(generated_text)
        if self.stop_tag in tail_text:
            self.stop_reason = "stop_tag"
            return True
        return False

    def _consume_skip_signal(self):
        if not os.path.exists(self.control_path):
            return False
        try:
            with open(self.control_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (json.JSONDecodeError, OSError):
            return False
        if not data.get("skip_current", False):
            return False
        data["skip_current"] = False
        try:
            with open(self.control_path, "w", encoding="utf-8") as handle:
                json.dump(data, handle)
        except OSError:
            pass
        return True

def validate_output(text, extractor):
    """Ensures generation didn't cut off and hooks fired successfully."""
    required_tags = ["<think>", "</think>", "<answer>", "</answer>", "<confidence>", "</confidence>"]
    for tag in required_tags:
        if tag not in text:
            return False
    if len(extractor.activations) == 0 or len(extractor.sparse_edges) == 0:
        return False
    return True
# ---------------------------------------------------

def build_cli_args():
    parser = argparse.ArgumentParser(description="Collect benchmark or custom traces.")
    parser.add_argument("--dataset", choices=["math", "nq"], default="math")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--custom-prompt", default=None)
    parser.add_argument("--custom-context", default="")
    parser.add_argument("--custom-gold-answer", default="")
    parser.add_argument("--custom-sample-id", default="custom_0")
    parser.add_argument("--custom-dataset-name", default="CustomUserPrompt")
    return parser.parse_args()


def maybe_build_custom_example(args):
    if not args.custom_prompt:
        return None
    return BenchmarkExample(
        sample_id=args.custom_sample_id,
        dataset_name=args.custom_dataset_name,
        split="custom",
        prompt=args.custom_prompt,
        gold_answer=args.custom_gold_answer,
        context=args.custom_context or None,
    )


def main():
    args = build_cli_args()
    print("🚀 Initializing CHARM Automated Benchmark Collector...")
    
    # 1. Initialize Generator (Qwen + Outlines DFA Strict Fence)
    hf_model, generator, tokenizer = build_generator("Qwen/Qwen3.5-0.8B")
    hf_model.config.output_attentions = True
    live_bus = LiveEventBus()
    live_reporter = LiveDemoReporter(live_bus)
    stopper = StopOnTag(tokenizer, reporter=live_reporter)
    halt_state = StoppingCriteriaList([stopper])
    
    # 2. Initialize Core Modules
    extractor = TraceExtractor(threshold=0.05)
    extractor.set_reporter(live_reporter)
    extractor.attach_hooks(hf_model) # Defaults to last layer
    
    loader = BenchmarkLoader()
    labeler = AutoLabeler()
    writer = TraceDatasetManager(save_dir="charm_unified_dataset")
    
    custom_example = maybe_build_custom_example(args)
    if custom_example is not None:
        test_subset = [custom_example]
        active_dataset_name = args.custom_dataset_name
        print("\n📚 Running a single custom prompt sample...")
    else:
        if args.dataset == "nq":
            print(f"\n📚 Loading NaturalQuestions ({args.split})...")
            dataset = loader.load_nq(split=args.split)
            active_dataset_name = "NaturalQuestions"
        else:
            print(f"\n📚 Loading AnswerableMath ({args.split})...")
            dataset = loader.load_math(split=args.split)
            active_dataset_name = "AnswerableMath"

        if not dataset:
            print("❌ Dataset not found. Ensure required files exist under 'tokenizer/data/llmsknow'.")
            return
        test_subset = dataset[: args.limit]

    live_reporter.start_run(
        model_name="Qwen/Qwen3.5-0.8B",
        dataset_name=active_dataset_name,
        total_samples=len(test_subset),
    )
    print(f"🎯 Starting automated collection for {len(test_subset)} samples.\n")
    
    # 4. The Automated Generation & Extraction Loop
    for ex in tqdm(test_subset, desc="Generating Traces"):
        extractor.clear()
        
        # Build Prompt
        system_prompt = "You are a logical reasoning assistant. You must rigorously follow this format:\n<think>\n[Your step-by-step reasoning]\n</think>\n<answer>\n[Your final short answer]\n</answer>\n<confidence>\n[A number between 0.0 and 1.0]\n</confidence>"
        
        user_content = ex.prompt
        if ex.context: # If it's a RAG/Summarization task, inject context
            user_content = f"Context: {ex.context}\n\nQuestion: {ex.prompt}"
            
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Tell the extractor where the prompt ends (Needed for Lookback Ratio math)
        prompt_token_count = len(tokenizer.encode(prompt))
        extractor.set_context_length(prompt_token_count)
        stopper.configure_sample(prompt_token_count)
        live_reporter.start_sample(ex, prompt, prompt_token_count)
        
        # Generate Trace
        try:
            result = generator(prompt, stopping_criteria=halt_state)
        except Exception as e:
            print(f"\n⚠️ Generation failed for {ex.sample_id}: {e}")
            live_reporter.mark_sample_failed(ex.sample_id, str(e))
            continue

        if stopper.stop_reason == "manual_skip":
            print(f"\n⏭️ Skipped {ex.sample_id} from UI control.")
            live_reporter.mark_sample_failed(ex.sample_id, "Skipped from frontend control.")
            continue

        if stopper.stop_reason == "max_generated_tokens":
            print(f"\n⚠️ Early stop on {ex.sample_id}: exceeded generated token limit.")
            live_reporter.mark_sample_failed(ex.sample_id, "Early stop: max generated token limit.")
            continue
            
        # Validate Structure
        if not validate_output(result, extractor):
            print(f"\n⚠️ Trace validation failed for {ex.sample_id} (Likely Hit Token Limit). Skipping.")
            live_reporter.mark_sample_failed(ex.sample_id, "Trace validation failed.")
            continue
            
        # Auto-Label Correctness (The "Oracle")
        labels_dict = labeler.get_labels(result, ex.gold_answer, ex.dataset_name)
        exact_ans = labels_dict.pop("exact_answer_extracted") # Pull out text string
        
        # Package into our Schema
        sample = UnifiedTraceRecord(
            sample_id=ex.sample_id,
            source_dataset=ex.dataset_name,
            prompt=ex.prompt,
            context=ex.context,
            gold_answer=ex.gold_answer,
            generated_text=result,
            exact_answer_extracted=exact_ans,
            labels={"gold_correctness": labels_dict.get("correctness", 0)},
            activations=extractor.activations.copy(),
            sparse_edges=extractor.sparse_edges.copy(),
            lookback_ratios=extractor.lookback_ratios.copy(),
            metadata={"prompt_tokens": prompt_token_count}
        )
        
        # Save to disk as .pt file
        save_path = writer.save_unified_trace(sample)
        live_reporter.mark_sample_saved(
            ex.sample_id,
            save_path,
            sample.labels.get("gold_correctness", 0),
        )
        
    print("\n✅ Benchmark Collection Complete!")
    extractor.remove_hooks()
    live_reporter.finish_run()

if __name__ == "__main__":
    main()