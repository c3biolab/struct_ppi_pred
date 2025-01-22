import torch
import torch.nn as nn
import torch.nn.functional as F

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
    Implements Cross-Attention mechanism.

    Attributes:
        embedding_dim (int): Dimensionality of the input embeddings.
        num_heads (int): Number of attention heads.
    """
    def __init__(self, embedding_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        assert self.head_dim * num_heads == embedding_dim, "embedding_dim must be divisible by num_heads"

        self.q_linear = nn.Linear(embedding_dim, embedding_dim)
        self.k_linear = nn.Linear(embedding_dim, embedding_dim)
        self.v_linear = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)


    def forward(self, query, key, value):
      """
      Perform cross-attention between query, key and value.
      Args:
          query (torch.Tensor): Query tensor.
          key (torch.Tensor): Key tensor.
          value (torch.Tensor): Value tensor.

      Returns:
          torch.Tensor: Weighted value tensor based on attention scores.
      """

      batch_size = query.size(0)
      q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2) # [batch_size, num_heads, seq_len, head_dim]
      k = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)    
      v = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)  
      
      attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [batch_size, num_heads, seq_len_q, seq_len_k]
      attn_weights = F.softmax(attn_scores, dim=-1)

      output = torch.matmul(attn_weights, v)                                    # [batch_size, num_heads, seq_len_q, head_dim]
      output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.embedding_dim)
      output = self.out_proj(output)                                             # [batch_size, seq_len, embedding_dim]

      return output.mean(dim=1, keepdim=False)



class BiDirectionalCrossAttention(nn.Module):
    """
    Implements Bi-directional Cross-Attention mechanism for combining embeddings.

    Attributes:
        embedding_dim (int): Dimensionality of the input embeddings.
        num_heads (int): Number of attention heads.
    """

    def __init__(self, embedding_dim, num_heads):
        super(BiDirectionalCrossAttention, self).__init__()
        self.cross_attention_p1_to_p2 = CrossAttention(embedding_dim, num_heads)
        self.cross_attention_p2_to_p1 = CrossAttention(embedding_dim, num_heads)
        self.fusion_proj = nn.Linear(embedding_dim * 2, embedding_dim)

    def forward(self, p1_embed, p2_embed):
      """
      Performs bi-directional cross-attention fusion between two input embeddings.

      Args:
        p1_embed (torch.Tensor): Embedding of protein 1.
        p2_embed (torch.Tensor): Embedding of protein 2.

      Returns:
        torch.Tensor: Fused embedding.
      """
      
      p1_to_p2 = self.cross_attention_p1_to_p2(p1_embed, p2_embed, p2_embed) # p1 attends to p2
      p2_to_p1 = self.cross_attention_p2_to_p1(p2_embed, p1_embed, p1_embed) # p2 attends to p1

      # Combine the original embeddings with the attention weighted embeddings
      fused_p1 = p1_embed + p1_to_p2
      fused_p2 = p2_embed + p2_to_p1
      
      # Concatenate and project to have a single embedding 
      fused_embedding = self.fusion_proj(torch.cat([fused_p1, fused_p2], dim=-1))

      return fused_embedding

class PPI_Model(nn.Module):
    def __init__(self): 
        super(PPI_Model, self).__init__()
        embedding_dim = 512

        self.projection_dim = 256
        self.heads = 4
        self.p1_projection = nn.Linear(embedding_dim, self.projection_dim)
        self.p2_projection = nn.Linear(embedding_dim, self.projection_dim)
        self.cross_attention = BiDirectionalCrossAttention(self.projection_dim, 4)
        self.fc1 = nn.Linear(self.projection_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, p1_embed, p2_embed):
     
        p1_embed = self.relu(self.p1_projection(p1_embed))
        p2_embed = self.relu(self.p2_projection(p2_embed))

        # Bi-directional Cross-Attention
        x = self.cross_attention(p1_embed, p2_embed)

        x = self.dropout1(x)
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x