# --- dataset_manager.py ---
import os
import torch
from torch_geometric.data import Data
from storage.schema import UnifiedTraceRecord

class TraceDatasetManager:
    def __init__(self, save_dir="charm_unified_dataset"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.counter = len(os.listdir(self.save_dir))

    def save_unified_trace(self, record: UnifiedTraceRecord):
        """
        Converts the UnifiedTraceRecord into a rich PyG Data object and saves it.
        """
        if not record.activations:
            print("⚠️ Trace buffer empty, skipping save.")
            return None

        print(f"\n📦 Packaging Unified Trace [{record.source_dataset}]...")
        
        # 1. Node Features (Layer 24 Hidden States)
        x = torch.stack(record.activations)

        # 2. Edge Index & Attributes (Sparse Attention)
        source_nodes = []
        target_nodes = []
        edge_weights = []

        for step_edges in record.sparse_edges:
            for src, tgt, weight in step_edges:
                source_nodes.append(src)
                target_nodes.append(tgt)
                edge_weights.append([weight])

        if not source_nodes: 
             edge_index = torch.empty((2, 0), dtype=torch.long)
             edge_attr = torch.empty((0, 1), dtype=torch.float32)
        else:
            edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
            edge_attr = torch.tensor(edge_weights, dtype=torch.float32)

        # 3. Auxiliary Node Features (Lookback Ratios from Lookback-Lens)
        # If lookback ratios were captured, convert to tensor, else use empty
        if record.lookback_ratios:
            lookback_tensor = torch.tensor(record.lookback_ratios, dtype=torch.float32).unsqueeze(1)
        else:
            lookback_tensor = torch.zeros((x.shape[0], 1), dtype=torch.float32)

        # 4. Create the PyTorch Geometric Data object
        graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        # Attach auxiliary dense features
        graph_data.lookback_ratio = lookback_tensor
        
        # 5. Attach Tiered Labels (From LLMsKnow)
        # We store labels as a dictionary inside the PyG object so we can do multi-task learning later
        graph_data.y_correctness = torch.tensor([record.labels.get('gold_correctness', -1)], dtype=torch.float32)
        graph_data.y_human = torch.tensor([record.labels.get('human_hallucination', -1)], dtype=torch.float32)
        
        # 6. Attach Metadata (For debugging and provenance)
        graph_data.text_meta = {
            "sample_id": record.sample_id,
            "source_dataset": record.source_dataset,
            "gold_answer": record.gold_answer,
            "exact_answer_extracted": record.exact_answer_extracted,
            "generated_text": record.generated_text
        }

        # 7. Serialize and save to disk
        filepath = os.path.join(self.save_dir, f"sample_{str(self.counter).zfill(6)}.pt")
        torch.save(graph_data, filepath)
        
        print(f"💾 Saved {filepath} | Nodes: {graph_data.num_nodes} | Edges: {graph_data.num_edges} | Correctness: {graph_data.y_correctness.item()}")
        self.counter += 1
        
        return filepath