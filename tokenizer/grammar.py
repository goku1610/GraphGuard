import outlines
from outlines.types import Regex
from transformers import AutoModelForCausalLM, AutoTokenizer

def build_generator(model_name="Qwen/Qwen3.5-0.8B"):
    # Load model and tokenizer
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto",
        attn_implementation="eager" 
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Wrap model
    model = outlines.from_transformers(hf_model, tokenizer)
    # THE STRICT FENCE REGEX
    # [^<]+ guarantees no text can leak because the FSM triggers the moment it sees a '<'
    regex_pattern = r"<think>\n[^<]+</think>\n<answer>\n[^<]+</answer>\n<confidence>\n(?:0\.\d+|1\.0)\n</confidence>"
    # UPDATED REGEX: More robust handling of multiline text without relying on non-greedy matchers
    # regex_pattern = r"<think>\n(?:.|\n)+\n</think>\n<answer>\n(?:.|\n)+\n</answer>\n<confidence>\n(?:0\.\d+|1\.0)\n</confidence>"
    output_type = Regex(regex_pattern)
    
    # 4. Create a closure to maintain compatibility
    # ADDED: **kwargs to accept the stopping_criteria from main.py
    # 4. Create a closure to maintain compatibility
    # CHANGE: Bump max_new_tokens to 8192 (Qwen's absolute max context window)
    def generator(prompt, max_new_tokens=8192, **kwargs):
        return model(prompt, output_type, max_new_tokens=max_new_tokens, **kwargs)
    
    return hf_model, generator, tokenizer