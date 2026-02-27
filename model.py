"""
SwinEff-DR: Hybrid Swin Transformer & EfficientNet Architecture
for Multi-Scale Diabetic Retinopathy Detection.

Reconstructed from trained state dict keys.
- EfficientNet branch: torchvision efficientnet_b4 (output 1792-dim)
- Swin Transformer branch: custom Swin-B (embed_dim=128, depths=[2,2,18,2],
  num_heads=[4,8,16,32], window_size=7, output 1024-dim)
- Attention-based fusion → 5-class DR classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ═══════════════════ Custom Swin Transformer (matches checkpoint keys) ═══════

class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=7):
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads))
        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size), indexing='ij'))
        coords_flat = torch.flatten(coords, 1)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("relative_position_index", rel.sum(-1))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size ** 2, self.window_size ** 2, -1).permute(2, 0, 1).contiguous()
        attn = attn + bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = F.softmax(attn, dim=-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(B_, N, C))


class Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, dim * 4)

    @staticmethod
    def _partition(x, ws):
        B, H, W, C = x.shape
        return x.view(B, H // ws, ws, W // ws, ws, C).permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws * ws, C)

    @staticmethod
    def _reverse(w, ws, H, W):
        B = int(w.shape[0] / (H * W / ws / ws))
        return w.view(B, H // ws, W // ws, ws, ws, -1).permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)

    def _mask(self, H, W, device):
        if self.shift_size == 0: return None
        m = torch.zeros((1, H, W, 1), device=device)
        cnt = 0
        for hs in (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)):
            for ws in (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)):
                m[:, hs, ws, :] = cnt; cnt += 1
        mw = self._partition(m, self.window_size).squeeze(-1)
        return (mw.unsqueeze(1) - mw.unsqueeze(2)).masked_fill_(lambda x: x != 0, -100.0).masked_fill_(lambda x: x == 0, 0.0)

    def forward(self, x):
        B, H, W, C = x.shape
        sc = x; x = self.norm1(x)
        if self.shift_size > 0:
            x = torch.roll(x, (-self.shift_size, -self.shift_size), (1, 2))
        xw = self._partition(x, self.window_size)
        # Build mask
        mask = None
        if self.shift_size > 0:
            img_mask = torch.zeros((1, H, W, 1), device=x.device)
            cnt = 0
            for hs in (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)):
                for ws in (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)):
                    img_mask[:, hs, ws, :] = cnt; cnt += 1
            mw = self._partition(img_mask, self.window_size).squeeze(-1)
            mask = (mw.unsqueeze(1) - mw.unsqueeze(2))
            mask = mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)
        aw = self.attn(xw, mask)
        x = self._reverse(aw, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(x, (self.shift_size, self.shift_size), (1, 2))
        return (sc + x) + self.mlp(self.norm2(sc + x))


class PatchMerging(nn.Module):
    """2×2 spatial merge: concat 4 patches → norm(4*dim) → reduce(4*dim, 2*dim)"""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
    def forward(self, x):
        B, H, W, C = x.shape
        x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2],
                        x[:, 0::2, 1::2], x[:, 1::2, 1::2]], dim=-1)
        return self.reduction(self.norm(x))


class _SwinStage(nn.Module):
    def __init__(self, dim, depth, num_heads, ws=7, downsample_input_dim=None):
        super().__init__()
        self.downsample = PatchMerging(downsample_input_dim) if downsample_input_dim else None
        self.blocks = nn.ModuleList([
            SwinBlock(dim, num_heads, ws, 0 if i % 2 == 0 else ws // 2)
            for i in range(depth)])
    def forward(self, x):
        if self.downsample: x = self.downsample(x)
        for b in self.blocks: x = b(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, embed_dim=128, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        return self.norm(self.proj(x).permute(0, 2, 3, 1))


class CustomSwinB(nn.Module):
    """Swin-B matching checkpoint keys:
       patch_embed.*, layers.{0-3}.blocks.*, layers.{1-3}.downsample.*, norm.*"""
    def __init__(self):
        super().__init__()
        dims = [128, 256, 512, 1024]
        depths = [2, 2, 18, 2]
        heads = [4, 8, 16, 32]
        self.patch_embed = PatchEmbed(128, 4)
        self.layers = nn.ModuleList([
            _SwinStage(dims[0], depths[0], heads[0], 7, None),
            _SwinStage(dims[1], depths[1], heads[1], 7, dims[0]),
            _SwinStage(dims[2], depths[2], heads[2], 7, dims[1]),
            _SwinStage(dims[3], depths[3], heads[3], 7, dims[2]),
        ])
        self.norm = nn.LayerNorm(dims[-1])

    def forward(self, x):
        x = self.patch_embed(x)
        for layer in self.layers: x = layer(x)
        return self.norm(x).mean(dim=[1, 2])


# ═══════════════════ EfficientNet Wrapper ═══════════════════

class _EfficientNetWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = models.efficientnet_b4(weights=None).features
    def forward(self, x):
        return self.features(x)


# ═══════════════════ Full SwinEff-DR Model ═══════════════════

class SwinEffDR(nn.Module):
    def __init__(self, num_classes=5, dropout=0.3):
        super().__init__()
        self.efficientnet = _EfficientNetWrapper()
        self.swin = CustomSwinB()

        self.eff_proj = nn.Sequential(
            nn.Linear(1792, 512), nn.BatchNorm1d(512), nn.ReLU(True),
            nn.Dropout(dropout), nn.Linear(512, 256))
        self.swin_proj = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(True),
            nn.Dropout(dropout), nn.Linear(512, 256))

        self.attention_local = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(True), nn.Linear(128, 1))
        self.attention_global = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(True), nn.Linear(128, 1))

        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(True), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(True), nn.Dropout(dropout),
            nn.Linear(128, num_classes))

    def forward(self, x):
        ef = F.adaptive_avg_pool2d(self.efficientnet(x), 1).flatten(1)
        ef = self.eff_proj(ef)
        sf = self.swin_proj(self.swin(x))
        w = torch.softmax(torch.cat([
            self.attention_local(ef), self.attention_global(sf)], 1), 1)
        return self.classifier(torch.cat([w[:, 0:1]*ef, w[:, 1:2]*sf], 1))
