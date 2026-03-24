# --- tokenizer/extractors/trace_extractor.py ---
import torch

class TraceExtractor:
    """
    Surgically attaches to the LLM to extract Hidden States, Sparse Attention Graphs, 
    and Lookback-Lens Attention Ratios.
    """
    def __init__(self, threshold=0.05):
        self.threshold = threshold
        self.context_length = None  # Must be set before generation for Lookback-Lens
        self.reporter = None
        
        # Data Buffers
        self.sparse_edges = []
        self.activations = []
        self.lookback_ratios = []
        
        # PyTorch Hook Handles
        self._handles = []
        self._pending_step = None
        self._pending_edges = []
        self._pending_lookback_ratio = None

    def clear(self):
        """Resets the buffers for the next generation trace."""
        self.sparse_edges.clear()
        self.activations.clear()
        self.lookback_ratios.clear()
        self.context_length = None
        self._pending_step = None
        self._pending_edges = []
        self._pending_lookback_ratio = None

    def set_context_length(self, length: int):
        """Informs the extractor where the prompt context ends and the generation begins."""
        self.context_length = length

    def set_reporter(self, reporter):
        self.reporter = reporter

    def attach_hooks(self, model, layer_idx=-1):
        """Attaches forward hooks to the specified layer (default: last layer)."""
        target_layer = model.model.layers[layer_idx]
        self._handles.append(target_layer.register_forward_hook(self.activation_hook))
        self._handles.append(target_layer.self_attn.register_forward_hook(self.attention_hook))

    def remove_hooks(self):
        """Removes the hooks to prevent memory leaks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def activation_hook(self, module, input, output):
        """Captures the deep residual stream activation for the current token."""
        if self._pending_step is None:
            return

        # Safely extract hidden states whether wrapped in a tuple or not
        hidden_states = output[0] if isinstance(output, tuple) else output
        
        # Dynamically handle dimensions (Prefill 3D vs Decode 2D)
        if hidden_states.dim() == 3:
            # Shape: [batch, seq_len, hidden_dim]
            latest_token_activation = hidden_states[0, -1, :].detach().cpu()
        elif hidden_states.dim() == 2:
            # Shape: [batch_or_seq, hidden_dim]
            latest_token_activation = hidden_states[-1, :].detach().cpu()
        else:
            # Fallback for unexpected flattening
            latest_token_activation = hidden_states.view(-1).detach().cpu()
            
        self.activations.append(latest_token_activation)
        self.sparse_edges.append(self._pending_edges)
        if self._pending_lookback_ratio is not None:
            self.lookback_ratios.append(self._pending_lookback_ratio)

        if self.reporter is not None:
            self.reporter.add_graph_step(
                step_index=self._pending_step,
                edges=self._pending_edges,
                lookback_ratio=self._pending_lookback_ratio,
            )

        self._pending_step = None
        self._pending_edges = []
        self._pending_lookback_ratio = None

    def attention_hook(self, module, input, output):
        """Captures the Sparse Graph and computes the Lookback Ratio."""
        self._pending_step = None
        self._pending_edges = []
        self._pending_lookback_ratio = None

        if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
            attn_weights = output[1].detach() # Shape: [batch, heads, q_len, kv_len]
            
            if attn_weights.dim() == 4:
                attn_matrix = attn_weights[0]
            elif attn_weights.dim() == 3:
                attn_matrix = attn_weights
            else:
                return

            # Average across heads to get a single 2D adjacency matrix: [q_len, kv_len]
            avg_attn = attn_matrix.mean(dim=0).squeeze()
            q_len = attn_matrix.shape[-2]
            kv_len = attn_matrix.shape[-1]

            if self.context_length is None or q_len != 1 or kv_len <= self.context_length:
                return

            attn_vector = avg_attn[-1] if avg_attn.dim() > 1 else avg_attn
            generated_kv_len = kv_len - self.context_length
            if generated_kv_len <= 0:
                return

            current_step = generated_kv_len - 1
            generated_attn = attn_vector[self.context_length:]
            prior_generated_attn = generated_attn[:current_step]

            step_edges = []
            if prior_generated_attn.numel() > 0:
                indices = (prior_generated_attn > self.threshold).nonzero(as_tuple=False).view(-1)
                for idx in indices:
                    step_edges.append((current_step, idx.item(), prior_generated_attn[idx].item()))

            attn_on_context = attn_vector[:self.context_length].sum()
            attn_on_generated = generated_attn.sum()
            total_attn = attn_on_context + attn_on_generated
            ratio = (attn_on_context / total_attn).item() if total_attn > 0 else 0.0

            self._pending_step = current_step
            self._pending_edges = step_edges
            self._pending_lookback_ratio = ratio