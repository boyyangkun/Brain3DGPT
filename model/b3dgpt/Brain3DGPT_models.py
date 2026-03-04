import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class AnomalyEvidenceEncoder(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.meta_net = nn.Sequential(
            nn.Conv3d(dim_in, dim_in * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # D,H,W 缩小一半

            nn.Conv3d(dim_in * 4, dim_in * 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(dim_in * 16, dim_in * 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(dim_in * 64, dim_in * 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(dim_in * 256, dim_in * 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
        )

    def forward(self, x):
        img_prompts = self.meta_net(x)  # (B, dim_out, D, H, W)
        return img_prompts

class AnomalyHypothesisProposal(nn.Module):
    def __init__(self, topk: int):
        super().__init__()
        self.topk = topk

    def forward(self, feat, anomaly_map):
        """
        feat: (B, C, D, H, W)  anomaly evidence
        anomaly_map: (B, 1, D, H, W)
        """
        B, C, D, H, W = feat.shape

        scores = anomaly_map.flatten(2).squeeze(1)  # (B, N)
        feat_flat = feat.flatten(2).transpose(1, 2)  # (B, N, C)

        K = min(self.topk, feat_flat.size(1))
        idx = scores.topk(K, dim=1).indices  # (B, K)

        # 选出异常假设
        hypo_feat = torch.gather(
            feat_flat, 1, idx.unsqueeze(-1).expand(-1, -1, C)
        )
        hypo_score = torch.gather(scores, 1, idx)  # (B, K)

        return hypo_feat, hypo_score

class AnomalyHypothesisProjector(nn.Module):
    def __init__(self, dim_hidden, dim_out, topk=32):
        super().__init__()
        self.encoder = AnomalyEvidenceEncoder(1, dim_hidden)
        self.proposal = AnomalyHypothesisProposal(topk)

        # hypothesis embedding
        self.embed = nn.Linear(dim_hidden+1, dim_out)
        self.embed1 = nn.Linear(dim_hidden, dim_out)

        # role embedding（关键）
        self.role_embed = nn.Parameter(torch.randn(1, 1, dim_out))

    def forward(self, anomaly_map):
        """
        anomaly_map: (B, 1, D, H, W)
        """
        feat = self.encoder(anomaly_map)
        A = F.interpolate(
            anomaly_map,
            size=feat.shape[-3:],   # (D', H', W')
            mode="trilinear",
            align_corners=False
        )

        hypo_feat, hypo_score = self.proposal(feat, A)

        # 拼接置信度
        hypo_score = hypo_score.unsqueeze(-1)  # (B, K, 1)
        hypo = torch.cat([hypo_feat, hypo_score], dim=-1)

        token = self.embed(hypo)
        token = token + self.role_embed
        B, C, D, H, W = feat.shape
        feat = feat.view(B, C, D*H*W).permute(0, 2, 1)  # (B, N, dim_out)，N = D'*H'*W'
        feat = self.embed1(feat)

        return feat, token  # (B, K, dim_out)

    
class Projector(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 3,
    ):
        super().__init__()

        # Step 2: 3D convolution
        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride
        )
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        x = self.conv3d(x)
        B, C, H, W, D = x.shape
        x = x.flatten(2)                    # (B, C', N')
        x = x.transpose(1, 2)
        x = self.norm(x)

        return x   

    
class AnomalyGuidedDFF(nn.Module):
    def __init__(self, dim):
        super().__init__()

        # 通道注意力（保持不变）
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv_atten = nn.Sequential(
            nn.Conv3d(dim * 2, dim * 2, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.conv_redu = nn.Conv3d(dim * 2, dim, kernel_size=1, bias=False)

        # anomaly map 轻微平滑（可选）
        self.anomaly_proj = nn.Conv3d(1, 1, kernel_size=1, bias=False)

    def forward(self, x, skip, anomaly_map):
        """
        x, skip: (B, C, H, W, D)
        anomaly_map: (B, 1, H, W, D), in [0,1]
        """
        # ---------- Channel fusion ----------
        fused = torch.cat([x, skip], dim=1)
        ch_att = self.conv_atten(self.avg_pool(fused))
        fused = fused * ch_att
        fused = self.conv_redu(fused)

        # ---------- Anomaly-guided spatial gate ----------
        A = torch.sigmoid(self.anomaly_proj(anomaly_map))

        # anomaly 高：偏向 skip
        # anomaly 低：偏向 x
        guided = (1 - A) * x + A * skip

        # 最终输出
        out = fused * A + guided * (1 - A)
        return out
    
class AnomalyEvidenceAggregator(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, anomaly_tokens, visual_tokens):
        """
        anomaly_tokens: (B, K, D)
        visual_tokens:  (B, N, D)
        """
        # anomaly token 作为 query
        attn_out, _ = self.cross_attn(
            query=anomaly_tokens,
            key=visual_tokens,
            value=visual_tokens
        )

        # residual + norm，保证稳定性
        anomaly_tokens = self.norm(anomaly_tokens + attn_out)
        return anomaly_tokens
    
class MultiScaleAnomalyGuidedFusion(nn.Module):
    def __init__(self, a_in, b_in, c_in, d_in, out_dim=4096):
        super().__init__()
        # 把低层映射到 512
        self.adapt1 = nn.Conv3d(a_in, d_in, kernel_size=1, bias=False)
        self.adapt2 = nn.Conv3d(b_in, d_in, kernel_size=1, bias=False)
        self.adapt3 = nn.Conv3d(c_in, d_in, kernel_size=1, bias=False)

        self.AGF1 = AnomalyGuidedDFF(d_in)
        self.AGF2 = AnomalyGuidedDFF(d_in)
        self.AGF3 = AnomalyGuidedDFF(d_in)

        # 最后 projector
        self.anomalyprojector = AnomalyHypothesisProjector(dim_hidden=1024, dim_out=out_dim, topk=32)
        self.projector = Projector(d_in,out_dim,2,2)
        self.evidence_agg = AnomalyEvidenceAggregator(out_dim)

    def forward(self, x1, x2, x3, x4, anomaly_map):
        # resize 到 x4 大小
        x1 = F.interpolate(x1,size=x4.shape[-3:],mode="trilinear",align_corners=False)
        x2 = F.interpolate(x2,size=x4.shape[-3:],mode="trilinear",align_corners=False)
        x3 = F.interpolate(x3,size=x4.shape[-3:],mode="trilinear",align_corners=False)
        anomalymap = F.interpolate(anomaly_map,size=x4.shape[-3:],mode="trilinear",align_corners=False)

        # 通道对齐
        x1 = self.adapt1(x1)
        x2 = self.adapt2(x2)
        x3 = self.adapt3(x3)

        # DFF 融合
        x = self.AGF1(x1, x2, anomalymap)
        x = self.AGF2(x, x3, anomalymap)
        x = self.AGF3(x, x4, anomalymap)  # [B, 512, 8, 10, 8]

        v = self.projector(x)
        d,b = self.anomalyprojector(anomaly_map)  # [B, N, out_dim]

        c = self.evidence_agg(b, v)
        a =  torch.cat([d, c], dim=1)
        # print(v.shape,a.shape)
        return v,a