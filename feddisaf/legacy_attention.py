"""
服务器端多头注意力机制 - FedDisAF的核心创新
在服务器聚合时使用多头注意力加权
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any
from torch.utils.data import DataLoader  # 添加这一行


class MultiHeadAttention(nn.Module):
    """多头注意力机制 - 用于服务器端客户端加权"""
    
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if num_heads == 0:
            self.is_identity = True
            self.embed_dim = embed_dim
            self.num_heads = 1
            self.head_dim = embed_dim
            self.scale = 1.0
            self.q_proj = nn.Identity()
            self.k_proj = nn.Identity()
            self.v_proj = nn.Identity()
            self.out_proj = nn.Identity()
            self.dropout = nn.Dropout(0.0)
            return
        
        self.is_identity = False
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, 
                value: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query: [batch, seq_len, embed_dim]
            key: [batch, seq_len, embed_dim]
            value: [batch, seq_len, embed_dim]
            
        Returns:
            output: [batch, seq_len, embed_dim]
        """
        if self.is_identity:
            return query

        batch_size = query.size(0)
        
        Q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        
        return attn_output


class ClientAttentionEncoder(nn.Module):
    """
    客户端特征编码器 - 用于提取客户端表示并计算注意力权重
    """
    
    def __init__(self, feature_dim: int = 512, hidden_dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.attention = MultiHeadAttention(hidden_dim, num_heads)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        self.weight_proj = nn.Linear(hidden_dim, 1)
        
    def forward(self, client_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算客户端注意力权重
        
        Args:
            client_features: [num_clients, feature_dim] 客户端特征
            
        Returns:
            attended_features: [num_clients, hidden_dim] 注意力增强后的特征
            attention_weights: [num_clients] 归一化的注意力权重
        """
        features = self.feature_proj(client_features)  # [num_clients, hidden_dim]
        
        features = features.unsqueeze(0)  # [1, num_clients, hidden_dim]
        attended = self.attention(features, features, features)  # [1, num_clients, hidden_dim]
        attended = self.layer_norm(features + attended)  # 残差连接
        attended = attended.squeeze(0)  # [num_clients, hidden_dim]
        
        weights = self.weight_proj(attended).squeeze(-1)  # [num_clients]
        attention_weights = F.softmax(weights, dim=0)  # 归一化
        
        return attended, attention_weights


class ServerAttentionAggregator:
    """
    服务器端多头注意力聚合器
    使用多头注意力机制计算客户端权重并聚合更新
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        self.num_heads = config.get('feddisaf_num_heads', 4)
        self.attention_dim = config.get('feddisaf_attention_dim', 256)
        self.feature_dim = config.get('feddisaf_feature_dim', 512)
        self.use_attention = config.get('feddisaf_enabled', True)

        if self.num_heads == 0:
            self.use_attention = False
        
        if self.use_attention:
            self.attention_encoder = ClientAttentionEncoder(
                feature_dim=self.feature_dim,
                hidden_dim=self.attention_dim,
                num_heads=self.num_heads
            ).to(self.device)
        else:
            self.attention_encoder = None
        
        self.client_feature_history = {}
        
    def extract_client_features(self, client_model: nn.Module,
                              client_dataloader: DataLoader) -> Optional[torch.Tensor]:
        """提取客户端特征 (用于注意力计算)"""
        client_model.eval()
        features_list = []

        with torch.no_grad():
            for data, _ in client_dataloader:
                data = data.to(self.device)
                output = client_model(data)          # ★ 先不拆包

                if isinstance(output, (tuple, list)):
                    if len(output) == 3:
                        features = output[1]
                    elif len(output) == 2:
                        features = output[1]
                    else:
                        features = output[0]
                else:
                    features = output                # 直接返回 tensor 的模型

                features_list.append(features.detach().cpu())

                if len(features_list) * data.size(0) >= 100:
                    break

        if features_list:
            all_features = torch.cat(features_list, dim=0)
            return torch.mean(all_features, dim=0)
        return None
    
    def compute_attention_weights(self, client_features: List[torch.Tensor]) -> torch.Tensor:
        """
        使用多头注意力计算客户端权重
        
        Args:
            client_features: 客户端特征列表
            
        Returns:
            attention_weights: 归一化的注意力权重
        """
        if not self.use_attention or not client_features:
            return torch.ones(len(client_features)) / len(client_features)
        
        features = torch.stack([f.to(self.device) for f in client_features], dim=0)
        
        if features.size(-1) != self.feature_dim:
            proj = nn.Linear(features.size(-1), self.feature_dim).to(self.device)
            features = proj(features)
        
        _, attention_weights = self.attention_encoder(features)
        
        return attention_weights.detach()
