import os
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings

# Suppress sklearn convergence warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

def load_and_pool_data(dataset_dir):
    """
    Loads PyG graphs, ignores the edges, and applies Mean Pooling 
    to the token activations to create a standard 1D feature vector per sample.
    """
    print(f"Loading dataset from: {dataset_dir}")
    X_features = []
    y_labels = []

    if not os.path.exists(dataset_dir):
        print(f"Error: Directory {dataset_dir} does not exist.")
        return None, None

    valid_files = [f for f in os.listdir(dataset_dir) if f.endswith(".pt")]
    
    for filename in valid_files:
        filepath = os.path.join(dataset_dir, filename)
        try:
            # Load the PyG Data object
            graph_data = torch.load(filepath, weights_only=False)
            
            # MEAN POOLING: Collapse [num_tokens, 1024] -> [1024]
            node_features = graph_data.x
            pooled_vector = torch.mean(node_features, dim=0).to(torch.float32).numpy()
            
            # --- UPDATED: Advanced Schema Label Extraction ---
            label = None
            
            # 1. Check for manual human label first
            if getattr(graph_data, 'y_human', None) is not None and graph_data.y_human.item() != -1:
                label = int(graph_data.y_human.item())
                
            # 2. Fall back to the automated correctness grader
            elif getattr(graph_data, 'y_correctness', None) is not None and graph_data.y_correctness.item() != -1:
                correctness = int(graph_data.y_correctness.item())
                # If it's correct (1), Hallucination is 0. If incorrect (0), Hallucination is 1.
                label = 0 if correctness == 1 else 1
                
            # 3. Reject if missing
            if label is None:
                print(f"Missing valid label in {filename}! Skipping...")
                continue
            # -------------------------------------------------
            
            X_features.append(pooled_vector)
            y_labels.append(label)
            
        except Exception as e:
            print(f"Skipping {filename}: {e}")

    print(f"Successfully loaded {len(X_features)} samples.")
    return np.array(X_features), np.array(y_labels)

def train_baseline():
    # 1. Path to your compiled dataset
    dataset_dir = os.path.join(os.path.dirname(__file__), "..", "charm_unified_dataset")
    
    # 2. Extract and Pool Features
    X, y = load_and_pool_data(dataset_dir)
    
    if X is None or len(X) < 10:
        print("Not enough data to train. Please generate at least 10 samples.")
        return

    # Check class balance
    num_hallucinations = sum(y)
    num_faithful = len(y) - num_hallucinations
    print(f"Class Distribution: {num_faithful} Faithful (0), {num_hallucinations} Hallucinations (1)")

    # 3. Train/Test Split (80% training, 20% testing)
    # stratify=y ensures both sets have the same ratio of hallucinations
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if min(num_hallucinations, num_faithful) >= 2 else None
    )

    print("\nTraining Logistic Regression Baseline (No Attention Edges)...")
    
    # 4. Initialize and Train the Baseline Model
    # max_iter=1000 gives it plenty of time to find the optimal weights for 1024 dimensions
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train, y_train)

    # 5. Evaluate the Model
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print("="*50)
    print("🎯 BASELINE RESULTS (Mean Pooled Activations)")
    print("="*50)
    print(f"Test Accuracy: {acc * 100:.2f}%")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Faithful (0)", "Hallucination (1)"]))
    print("="*50)

if __name__ == "__main__":
    train_baseline()