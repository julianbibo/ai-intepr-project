from torch import nn
import torch


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.adapter = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout)
        )

        # Residual connection weight (learnable)
        self.residual_weight = nn.Parameter(
            torch.tensor(0.5), requires_grad=True)
        self.adapted_weight = nn.Parameter(
            torch.tensor(0.5), requires_grad=True)

    def forward(self, x):
        """
        Args:
            x: Input tensor
               - Training: [batch_size, seq_len, 1024] or [batch_size, 1024]
               - Inference: [2, seq_len, 1024] (both CFG paths)
        Returns:
            Output tensor with same shape as input
        """
        adapted = self.adapter(x)
        return x * self.residual_weight + adapted * self.adapted_weight

    @classmethod
    def from_args(cls, args):
        return cls(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            dropout=args.dropout
        )
