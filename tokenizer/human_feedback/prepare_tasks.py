import argparse
import json
import os
import sys
from dataclasses import asdict

from transformers import StoppingCriteria, StoppingCriteriaList

TOKENIZER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(TOKENIZER_DIR)

from grammar import build_generator
from datasets.benchmark_loader import BenchmarkLoader
from labels.auto_correctness import AutoLabeler


class StopOnTag(StoppingCriteria):
    def __init__(self, tokenizer, stop_tag="</confidence>", max_generated_tokens=1024, check_window=30):
        self.tokenizer = tokenizer
        self.stop_tag = stop_tag
        self.max_generated_tokens = max_generated_tokens
        self.check_window = check_window
        self.prompt_token_count = 0

    def configure_prompt(self, prompt_token_count: int):
        self.prompt_token_count = prompt_token_count

    def __call__(self, input_ids, scores, **kwargs):
        # Stop if tag appears in the tail (robust enough for stage output)
        tail_text = self.tokenizer.decode(input_ids[0][-self.check_window:])
        if self.stop_tag in tail_text:
            return True

        # Safety cap in case tag never appears
        if self.prompt_token_count:
            generated_ids = input_ids[0][self.prompt_token_count:]
            if generated_ids.shape[0] >= self.max_generated_tokens:
                return True

        return False


def parse_args():
    parser = argparse.ArgumentParser(description="Generate annotation tasks with model answers.")
    parser.add_argument(
        "--dataset",
        choices=["math", "nq", "movie_qa", "mnli", "winogrande", "winobias", "all"],
        default="math",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--output",
        default=os.path.join(TOKENIZER_DIR, "human_feedback", "tasks.jsonl"),
    )
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--print-full-generation", action="store_true")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete output file and start from scratch. Without --fresh, append and skip sample_ids already in the file.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-generated-tokens", type=int, default=1024)
    return parser.parse_args()


def build_prompt(tokenizer, example):
    system_prompt = (
        "You are a logical reasoning assistant. You must rigorously follow this format:\n"
        "<think>\n[Your step-by-step reasoning]\n</think>\n"
        "<answer>\n[Your final short answer]\n</answer>\n"
        "<confidence>\n[A number between 0.0 and 1.0]\n</confidence>"
    )
    user_content = example.prompt
    if example.context:
        user_content = f"Context: {example.context}\n\nQuestion: {example.prompt}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def validate_required_tags(text: str) -> bool:
    required_tags = ["<think>", "</think>", "<answer>", "</answer>", "<confidence>", "</confidence>"]
    return all(tag in text for tag in required_tags)


def clean_generation_text(text: str) -> str:
    cleaned = text
    for r in ["<|endoftext|>", "<|im_end|>", "<|end|>"]:
        cleaned = cleaned.replace(r, "")
    return cleaned.strip()


def load_done_sample_ids(output_path: str) -> set:
    done = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample = obj.get("sample") or {}
            sid = sample.get("sample_id") or obj.get("sample_id")
            if sid:
                done.add(sid)
    return done


def main():
    args = parse_args()
    loader = BenchmarkLoader()
    labeler = AutoLabeler()

    dataset_keys = ["math", "nq", "movie_qa", "mnli", "winogrande", "winobias"]
    if args.dataset == "all":
        dataset = []
        for key in dataset_keys:
            part = loader.load_dataset(dataset=key, split=args.split)
            if not part and key == "mnli":
                part = loader.load_dataset(dataset=key, split="train")
            if not part:
                print(f"skip dataset '{key}' (no rows found)")
                continue
            capped = part[: args.limit]
            print(f"loaded {len(capped)} from {key}")
            dataset.extend(capped)
    else:
        dataset = loader.load_dataset(dataset=args.dataset, split=args.split)
        if not dataset and args.dataset == "mnli":
            dataset = loader.load_dataset(dataset=args.dataset, split="train")
        dataset = dataset[: args.limit]
    examples = dataset
    if not examples:
        raise RuntimeError("No dataset examples found for requested dataset/split.")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    if args.fresh and os.path.exists(args.output):
        os.remove(args.output)

    done_sample_ids = load_done_sample_ids(args.output)
    if done_sample_ids:
        print(f"Resume: {len(done_sample_ids)} sample_id(s) already in {args.output}; will skip those.")

    pending = [ex for ex in examples if ex.sample_id not in done_sample_ids]
    skipped = len(examples) - len(pending)
    if skipped:
        print(f"Skipping {skipped} already-saved examples.")

    if not pending:
        print("Nothing to generate; all examples are already in the output file.")
        return

    model, generator, tokenizer = build_generator(args.model_name)
    model.eval()
    model.config.output_attentions = False
    halt_criteria = StopOnTag(
        tokenizer,
        max_generated_tokens=args.max_generated_tokens,
    )
    halt_state = StoppingCriteriaList([halt_criteria])

    written = 0
    file_mode = "a" if os.path.exists(args.output) else "w"
    with open(args.output, file_mode, encoding="utf-8") as handle:
        for run_idx, ex in enumerate(pending):
            prompt = build_prompt(tokenizer, ex)
            try:
                generated_text = clean_generation_text(
                    generator(prompt, stopping_criteria=halt_state)
                )
            except Exception as exc:
                print(f"skip {ex.sample_id}: generation error: {exc}")
                continue

            labels = labeler.get_labels(generated_text, ex.gold_answer, ex.dataset_name)
            structure_ok = validate_required_tags(generated_text)
            task = {
                "task_id": f"{ex.dataset_name}_{ex.split}_{ex.sample_id}",
                "sample": asdict(ex),
                "model_name": args.model_name,
                "generated_text": generated_text,
                "extracted_answer": labels.get("exact_answer_extracted", ""),
                "auto_correctness": labels.get("correctness", 0),
                "structure_ok": structure_ok,
            }
            handle.write(json.dumps(task, ensure_ascii=True) + "\n")
            handle.flush()
            written += 1

            extracted_answer = labels.get("exact_answer_extracted", "")
            auto_correctness = labels.get("correctness", 0)
            print("=" * 90)
            print(
                f"[+{written} this run | {run_idx + 1}/{len(pending)} pending] "
                f"sample_id: {ex.sample_id} | dataset: {ex.dataset_name} | structure_ok: {structure_ok}"
            )
            print(f"Question: {ex.prompt}")
            if ex.context:
                context_preview = ex.context if len(ex.context) <= 220 else ex.context[:220] + "...[truncated]"
                print(f"Context: {context_preview}")
            print(f"Gold Answer: {ex.gold_answer}")
            print(f"Extracted Answer: {extracted_answer}")
            print(f"Auto Correctness: {auto_correctness}")
            if args.print_full_generation:
                print("Model Output:")
                print(generated_text)
            else:
                output_preview = generated_text if len(generated_text) <= 320 else generated_text[:320] + "...[truncated]"
                print(f"Model Output Preview: {output_preview}")
            print("=" * 90)

    print(f"wrote {written} tasks to {args.output}")


if __name__ == "__main__":
    main()
