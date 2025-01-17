import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .MAPE_PPI.mape_enc import CodeBook

class FocalLoss(nn.Module):
    """
    Implements Focal Loss, which is useful for addressing class imbalance in binary classification tasks.

    Attributes:
        alpha (float): Scaling factor for the positive class.
        gamma (float): Focusing parameter to reduce the relative loss for well-classified examples.
    """
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        """
        Compute the Focal Loss.

        Args:
            inputs (torch.Tensor): Predicted logits.
            targets (torch.Tensor): Ground truth binary labels.

        Returns:
            torch.Tensor: Computed Focal Loss.
        """
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return torch.mean(F_loss)

class CrossAttention(nn.Module):
    """
    Implements Cross-Attention mechanism using multi-head attention.

    Attributes:
        embed_dim (int): Dimensionality of the input embeddings.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout rate for attention weights.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        """
        Apply cross-attention.

        Args:
            query (torch.Tensor): Query tensor.
            key (torch.Tensor): Key tensor.
            value (torch.Tensor): Value tensor.

        Returns:
            torch.Tensor: Output tensor after applying attention.
        """
        attn_output, _ = self.multihead_attn(query, key, value)
        output = self.norm(query + self.dropout(attn_output))  # Residual connection + normalization
        return output

class AdaptiveFusion(nn.Module):
    """
    Implements Adaptive Fusion mechanism for combining embeddings.

    Attributes:
        embedding_dim (int): Dimensionality of the input embeddings.
    """
    def __init__(self, embedding_dim):
        super(AdaptiveFusion, self).__init__()
        self.weight_layer = nn.Linear(embedding_dim * 2, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, p1_embed, p2_embed):
        """
        Perform adaptive fusion of two embeddings.

        Args:
            p1_embed (torch.Tensor): Embedding of protein 1.
            p2_embed (torch.Tensor): Embedding of protein 2.

        Returns:
            torch.Tensor: Fused embedding.
        """
        combined = torch.cat([p1_embed, p2_embed], dim=1)
        weights = self.softmax(self.weight_layer(combined))
        fused_embedding = weights[:, 0].unsqueeze(1) * p1_embed + weights[:, 1].unsqueeze(1) * p2_embed
        return fused_embedding

class PPI_Model(nn.Module):
    """
    Protein-Protein Interaction (PPI) Model leveraging embeddings from a pretrained VAE (MAPE) model,
    attention mechanisms, and fully connected layers.

    Attributes:
        vae_model (CodeBook): Pretrained MAPE model for protein embeddings.
        cross_attention (CrossAttention): Cross-attention layer for processing protein embeddings.
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        fc3 (nn.Linear): Third fully connected layer.
        fc4 (nn.Linear): Output layer.
        dropout (nn.Dropout): Dropout layer to prevent overfitting.
    """
    def __init__(self, mape_cfg, mape_weights_path):
        """
        Initialize the PPI_Model.

        Args:
            mape_cfg (dict): Configuration for the pretrained MAPE model.
            mape_weights_path (str): Path to the pretrained MAPE model weights.
        """
        super(PPI_Model, self).__init__()

        self.vae_model = CodeBook(mape_cfg)
        self.vae_model.load_state_dict(torch.load(mape_weights_path))
        self.vae_model = self.vae_model.to("cuda")

        # Freeze the VAE model
        for param in self.vae_model.parameters():
            param.requires_grad = False

        # Define the layers: input - output dimensions
        embedding_dim = 512

        self.cross_attention = CrossAttention(embedding_dim, num_heads=4, dropout=0.1)

        self.fc1 = nn.Linear(embedding_dim * 2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, p1, p2):
        """
        Forward pass of the PPI_Model.

        Args:
            p1 (torch.Tensor): Input features for protein 1.
            p2 (torch.Tensor): Input features for protein 2.

        Returns:
            torch.Tensor: Logits indicating the likelihood of interaction between the two proteins.
        """
        # Obtain protein embeddings
        p1_embed = self.vae_model.Protein_Encoder.forward(p1, self.vae_model.vq_layer)
        p2_embed = self.vae_model.Protein_Encoder.forward(p2, self.vae_model.vq_layer)

        if len(p1_embed.shape) == 2:
            p1_embed = p1_embed.unsqueeze(1)
        if len(p2_embed.shape) == 2:
            p2_embed = p2_embed.unsqueeze(1)

        # Move the embeddings to the same device
        p1_embed = p1_embed.to('cuda')
        p2_embed = p2_embed.to('cuda')

        # Apply attention mechanism
        attended_a = self.cross_attention(p1_embed, p2_embed, p2_embed)  # A attends to B
        attended_b = self.cross_attention(p2_embed, p1_embed, p1_embed)  # B attends to A

        # Pooling (mean pooling for simplicity)
        pooled_a = torch.mean(attended_a, dim=1)
        pooled_b = torch.mean(attended_b, dim=1)

        # Concatenate pooled embeddings
        combined = torch.cat([pooled_a, pooled_b], dim=1)

        # Fully connected layers
        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)

        return x