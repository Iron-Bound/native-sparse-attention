import torch
import torch.nn as nn
from typing import List, Tuple

class TokenCompression(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, block_size: int):
        super().__init__()
        self.block_size = block_size
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.position_encoding = nn.Parameter(torch.randn(block_size, hidden_size))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape input into blocks
        B, T, H = x.size()
        x_blocks = x.view(B, -1, self.block_size, H)
        
        # Add position encoding
        pos_enc = self.position_encoding.unsqueeze(0).unsqueeze(0)
        x_blocks = x_blocks + pos_enc
        
        # Apply MLP to get compressed representation
        compressed = self.mlp(x_blocks.mean(dim=-2))
        return compressed

class TokenSelection(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, top_k: float):
        super().__init__()
        self.top_k = top_k
        self.importance_score_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, x: torch.Tensor, attention_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute block importance scores
        B, T, H = x.size()
        importance_scores = self.importance_score_mlp(x).squeeze(-1)
        
        # Get top-k blocks
        num_blocks = int(T * self.top_k)
        _, selected_indices = torch.topk(importance_scores, k=num_blocks)
        
        # Select tokens
        selected_tokens = x[torch.arange(B)[:, None], selected_indices]
        return selected_tokens, selected_indices

class SlidingWindowAttention(nn.Module):
    def __init__(self, window_size: int, hidden_size: int):
        super().__init__()
        self.window_size = window_size
        self.query_linear = nn.Linear(hidden_size, hidden_size)
        self.key_linear = nn.Linear(hidden_size, hidden_size)
        self.value_linear = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, T, H = q.size()
        outputs = []
        
        for i in range(0, T, self.window_size):
            start = max(0, i - self.window_size // 2)
            end = min(T, i + self.window_size // 2 + 1)
            
            window_q = q[:, i:i+1]
            window_k = k[:, start:end]
            window_v = v[:, start:end]
            
            # Compute attention within window
            Q = self.query_linear(window_q)
            K = self.key_linear(window_k)
            V = self.value_linear(window_v)
            
            scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(H)
            weights = nn.functional.softmax(scores, dim=-1)
            outputs.append(torch.matmul(weights, V))
        
        return torch.cat(outputs, dim=1)

class NativeSparseAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, 
                 block_size: int, top_k: float, window_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # Initialize three attention paths
        self.compression_path = TokenCompression(hidden_size, num_heads, block_size)
        self.selection_path = TokenSelection(hidden_size, num_heads, top_k)
        self.sliding_window_path = SlidingWindowAttention(window_size, hidden_size)
        
        # Gate scores MLP
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3),
            nn.Sigmoid()
        )
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        # Compute gate scores
        gate_scores = self.gate_mlp(query)
        g_comp, g_sel, g_win = gate_scores.chunk(3, dim=-1)
        
        # Process through each path
        compressed_output = self.compression_path(key)
        selected_tokens, _ = self.selection_path(key, None)
        window_output = self.sliding_window_path(query, key, value)
        
        # Combine outputs based on gate scores
        final_output = (
            g_comp * compressed_output +
            g_sel * selected_tokens +
            g_win * window_output
        )
        
        return final_output