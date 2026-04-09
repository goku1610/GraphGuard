import os
import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. The Neural Network Architecture
# ==========================================
class CHARMCritic(torch.nn.Module):
    def __init__(self, node_dim=1025, hidden_dim=256, edge_dim=1):
        super(CHARMCritic, self).__init__()
        self.conv1 = GATConv(node_dim, hidden_dim, heads=4, concat=False, edge_dim=edge_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False, edge_dim=edge_dim)
        self.lin1 = torch.nn.Linear(hidden_dim, 64)
        self.lin2 = torch.nn.Linear(64, 1)

    def forward(self, data):
        # Unpack the graph components
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Step 1: Message Passing (Tokens communicate along attention edges)
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        
        # Step 2: Readout (Collapse the entire graph into one vector)
        x = global_mean_pool(x, batch)
        
        # Step 3: Classification (0.0 to 1.0 Hallucination Score)
        x = F.relu(self.lin1(x))
        x = torch.sigmoid(self.lin2(x))
        return x

# ==========================================
# 2. Data Loading & Schema Handling
# ==========================================
def load_graph_dataset(dataset_dir, mode="att+act"):
    print(f"Loading PyG graphs from: {dataset_dir}  [mode={mode}]")
    dataset = []

    valid_files = [f for f in os.listdir(dataset_dir) if f.endswith(".pt")]
    
    for filename in valid_files:
        filepath = os.path.join(dataset_dir, filename)
        try:
            data = torch.load(filepath, weights_only=False)
            data.x = data.x.to(torch.float32)
            data.edge_attr = data.edge_attr.to(torch.float32)

            lookback = getattr(data, 'lookback_ratio', None)
            if lookback is not None and lookback.shape[0] == data.x.shape[0]:
                lb = lookback.to(torch.float32)
            else:
                lb = torch.zeros(data.x.shape[0], 1, dtype=torch.float32)

            if mode == "att":
                data.x = lb
            else:
                data.x = torch.cat([data.x, lb], dim=-1)
            
            # --- THE FIX: Prune Ghost Edges ---
            # 1. Explicitly define the number of nodes
            num_actual_nodes = data.x.size(0)
            data.num_nodes = num_actual_nodes
            
            # 2. Find edges where both the Source and Target exist in our x tensor
            valid_edge_mask = (data.edge_index[0] < num_actual_nodes) & (data.edge_index[1] < num_actual_nodes)
            
            # 3. Filter the graph to only include valid edges
            data.edge_index = data.edge_index[:, valid_edge_mask]
            data.edge_attr = data.edge_attr[valid_edge_mask]
            # ----------------------------------

            # Label Schema Mapping (Same as Baseline)
            label = None
            if getattr(data, 'y_human', None) is not None and data.y_human.item() != -1:
                label = int(data.y_human.item())
            elif getattr(data, 'y_correctness', None) is not None and data.y_correctness.item() != -1:
                correctness = int(data.y_correctness.item())
                label = 0 if correctness == 1 else 1
                
            if label is None:
                continue
                
            # Assign the target tensor
            data.y = torch.tensor([label], dtype=torch.float32)
            dataset.append(data)
            
        except Exception as e:
            pass

    print(f"Successfully loaded {len(dataset)} valid graphs.")
    return dataset

# ==========================================
# 3. The Training Engine
# ==========================================
NODE_DIM = {"att": 1, "att+act": 1025}

def train_gnn(mode="att+act"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training CHARM GNN  [mode={mode}, device={device}]")
    
    dataset_dir = os.path.join(os.path.dirname(__file__), "..", "charm_unified_dataset")
    dataset = load_graph_dataset(dataset_dir, mode=mode)
    
    if len(dataset) < 10:
        print("Not enough data to train. Please generate more samples.")
        return

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    node_dim = NODE_DIM[mode]
    model = CHARMCritic(node_dim=node_dim, hidden_dim=256, edge_dim=1).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    epochs = 20
    print("="*50)
    print("🧠 STARTING MESSAGE PASSING TRAINING")
    print("="*50)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0

        # --- Training Loop ---
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward Pass
            out = model(batch).squeeze(-1)
            loss = criterion(out, batch.y)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch.num_graphs
            preds = (out > 0.5).float()
            correct_train += int((preds == batch.y).sum())
            total_train += batch.num_graphs

        # --- Validation Loop ---
        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch).squeeze(-1)
                preds = (out > 0.5).float()
                correct_test += int((preds == batch.y).sum())
                total_test += batch.num_graphs

        train_acc = (correct_train / total_train) * 100
        test_acc = (correct_test / total_test) * 100
        avg_loss = total_loss / total_train
        
        print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.1f}% | Test Acc: {test_acc:.1f}%")

    os.makedirs(os.path.join(os.path.dirname(__file__), "weights"), exist_ok=True)
    weight_name = f"charm_critic_{mode.replace('+', '_')}_v1.pth"
    save_path = os.path.join(os.path.dirname(__file__), "weights", weight_name)
    torch.save(model.state_dict(), save_path)
    
    print("="*50)
    print(f"GNN Training Complete! Weights saved to:\n{save_path}")
    print("="*50)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["att", "att+act"], default="att+act")
    args = parser.parse_args()
    train_gnn(mode=args.mode)