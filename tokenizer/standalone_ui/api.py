import os
import sys
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from torch_geometric.data import Data

# Safely point back to your core modules without moving them
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from grammar import build_generator
from extractors.trace_extractor import TraceExtractor
from trainers.gnn_trainer import CHARMCritic
from transformers import StoppingCriteria, StoppingCriteriaList

app = FastAPI(title="CHARM Standalone UI Engine")

# Crucial for React to communicate with this port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    prompt: str

# Global state
llm_model, llm_generator, llm_tokenizer = None, None, None
gnn_model, extractor, halt_state = None, None, None

class StopOnTag(StoppingCriteria):
    def __init__(self, tokenizer, stop_tag="</confidence>"):
        self.tokenizer = tokenizer
        self.stop_tag = stop_tag
    def __call__(self, input_ids, scores, **kwargs):
        return self.stop_tag in self.tokenizer.decode(input_ids[0][-15:])

@app.on_event("startup")
def load_models():
    global llm_model, llm_generator, llm_tokenizer, gnn_model, extractor, halt_state
    print("🚀 Booting Independent AI Backend...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load LLM
    llm_model, llm_generator, llm_tokenizer = build_generator("Qwen/Qwen3.5-0.8B")
    llm_model.config.output_attentions = True
    halt_state = StoppingCriteriaList([StopOnTag(llm_tokenizer)])
    
    # Attach Extractor
    extractor = TraceExtractor(threshold=0.05)
    extractor.attach_hooks(llm_model)
    
    # Load your freshly trained GNN Brain
    gnn_model = CHARMCritic(node_dim=1024, hidden_dim=256).to(device)
    weights_path = os.path.join(PARENT_DIR, "trainers", "weights", "charm_critic_v1.pth")
    
    if os.path.exists(weights_path):
        gnn_model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        gnn_model.eval()
        print("🧠 CHARM Critic Weights Loaded Successfully!")
    else:
        print("⚠️ Warning: Weights not found at", weights_path)

@app.post("/api/analyze")
def analyze(req: GenerateRequest):
    global extractor, llm_tokenizer, llm_model
    extractor.clear()
    device = next(gnn_model.parameters()).device

    # 1. Formatting & Prompting
    system_prompt = "You are a logical reasoning assistant. You must rigorously follow this format:\n<think>\n[Your step-by-step reasoning]\n</think>\n<answer>\n[Your final short answer]\n</answer>\n<confidence>\n[A number between 0.0 and 1.0]\n</confidence>"
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": req.prompt}]
    full_prompt = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    prompt_ids = llm_tokenizer.encode(full_prompt, return_tensors="pt").to(llm_model.device)
    prompt_token_count = prompt_ids.shape[1]
    extractor.set_context_length(prompt_token_count)

    # 2. Generation with ID tracking
    print(f"\n📝 Analyzing: {req.prompt}")
    # We use generate directly to get the output IDs
    output_ids = llm_model.generate(
        prompt_ids, 
        max_new_tokens=512,
        stopping_criteria=halt_state,
        output_attentions=True,
        return_dict_in_generate=True
    )
    
    # Extract ONLY the newly generated tokens
    new_tokens_ids = output_ids.sequences[0][prompt_token_count:]
    # This is the "Truth" for our UI labels
    real_tokens = [llm_tokenizer.decode([tid]) for tid in new_tokens_ids]

    # 3. Graph Assembly (Using the Activation Count as the Source of Truth)
    x = torch.stack(extractor.activations).to(torch.float32)
    num_nodes = x.size(0)
    
    ui_edges, source_nodes, target_nodes, edge_weights = [], [], [], []
    for step_edges in extractor.sparse_edges:
        for src, tgt, weight in step_edges:
            if src < num_nodes and tgt < num_nodes:
                source_nodes.append(src)
                target_nodes.append(tgt)
                edge_weights.append([weight])
                ui_edges.append({"s": src, "t": tgt, "weight": str(round(weight, 2))})

    if not source_nodes:
        return {"error": "Graph empty."}

    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    edge_attr = torch.tensor(edge_weights, dtype=torch.float32)
    graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr).to(device)

    # 4. GNN Inference
    with torch.no_grad():
        response_score = gnn_model(graph_data).item()
        # Per-token calculation
        token_scores = [0.0] * num_nodes
        for tgt in target_nodes:
            token_scores[tgt] += 0.05 # Accumulate attention density
        
        # Blend with global response score to highlight "risky" areas
        token_scores = [min(1.0, (s * 0.5) + (response_score * 0.5)) for s in token_scores]

    # 5. UI Layout Mapping (Smarter Labels)
    ui_nodes = []
    for i in range(num_nodes):
        raw_label = real_tokens[i] if i < len(real_tokens) else f"T{i}"
        
        # Clean labels: remove tags and whitespace for the circle display
        clean_label = raw_label.replace("<", "").replace(">", "").replace("/", "").strip()
        if not clean_label: clean_label = "..." # Handle empty strings/whitespace
        
        ui_nodes.append({
            "id": i,
            "label": clean_label[:6], # Truncate for UI fit
            "x": 60 + ((i % 7) * 115), # Slightly wider grid
            "y": 60 + ((i // 7) * 100)
        })

    return {
        "tokens": real_tokens,
        "nodes": ui_nodes,
        "edges": ui_edges,
        "tokenScores": [round(s, 2) for s in token_scores],
        "responseScore": round(response_score, 2)
    }
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)