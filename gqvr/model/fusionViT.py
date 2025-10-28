import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# -------------------------
# Small building blocks
# -------------------------
class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# simplified multi-head attention (batch-first)
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner = dim_head * heads
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))
        self.scale = dim_head ** -0.5
    def forward(self, x, mask=None):
        # x: [B, L, D]
        B, L, D = x.shape
        qkv = self.to_qkv(x).view(B, L, 3, self.heads, self.dim_head)
        q, k, v = qkv[:,:,0], qkv[:,:,1], qkv[:,:,2]  # each [B, L, H, Dh]
        q = rearrange(q, 'b l h d -> b h l d')
        k = rearrange(k, 'b l h d -> b h l d')
        v = rearrange(v, 'b l h d -> b h l d')
        attn = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        if mask is not None:
            # mask shape [B, L] boolean keep mask
            maskq = mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,L]
            maskk = mask.unsqueeze(1).unsqueeze(3)  # [B,1,L,1]
            allowed = (maskq & maskk)
            attn = attn.masked_fill(~allowed, float('-1e9'))
        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h l d -> b l (h d)')
        return self.to_out(out)

class TransformerBlockSimple(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.attn = PreNorm(dim, MultiHeadSelfAttention(dim, heads, dim_head, dropout))
        self.ff = PreNorm(dim, MLP(dim, int(dim*mlp_ratio), dropout))
    def forward(self, x, mask=None):
        x = x + self.attn(x, mask=mask)
        x = x + self.ff(x)
        return x

# -------------------------
# Hybrid fusion module
# -------------------------
class LightweightHybrid3DFusion(nn.Module):
    """
    Hybrid 3D fusion:
      - Temporal attention per spatial patch (cheap: T small)
      - Spatial windowed attention per frame (cheap: windowed)
    Input: Z [B, T, C, H, W]
    Output: z_fused [B, C, H, W] (z_ref + residual)
    """
    def __init__(self, c_latent=4, patch_size=(2,4), window_size=(8,8),
                 embed_dim=512, t_heads=8, s_heads=8, depth_temporal=3, depth_spatial=2,
                 mlp_ratio=3.0, dropout=0.1, use_confidence=False):
        super().__init__()
        self.c = c_latent
        self.p_h, self.p_w = patch_size
        self.win_h, self.win_w = window_size
        self.embed_dim = embed_dim
        self.use_conf = use_confidence

        # projection of per-patch (per-frame) vector to embed
        patch_volume = self.c * (self.p_h * self.p_w)
        self.patch_proj = nn.Linear(patch_volume, embed_dim)
        self.patch_unproj = nn.Linear(embed_dim, patch_volume)

        # temporal transformer for per-patch tokens
        self.temporal_blocks = nn.ModuleList([TransformerBlockSimple(embed_dim, heads=t_heads, dim_head=embed_dim//t_heads,
                                                                      mlp_ratio=mlp_ratio, dropout=dropout)
                                              for _ in range(depth_temporal)])
        # spatial windowed attention blocks (applies inside each frame on windows of patches)
        self.spatial_blocks = nn.ModuleList([TransformerBlockSimple(embed_dim, heads=s_heads, dim_head=embed_dim//s_heads,
                                                                     mlp_ratio=mlp_ratio, dropout=dropout)
                                             for _ in range(depth_spatial)])

        # gating and final residual projection
        self.gate = nn.Sequential(nn.Linear(embed_dim, embed_dim//4), nn.GELU(),
                                  nn.Linear(embed_dim//4, 1), nn.Sigmoid())
        self.final_scale = nn.Parameter(torch.tensor(1.0))    # scale residual if needed
        self.temp_param = nn.Parameter(torch.tensor(0.1))     # initial temperature
        self.sigma_param = nn.Parameter(torch.tensor(2.0))    # initial Gaussian width
        

    def forward(self, Z, conf_mask=None):
        """
        Z: [B, T, C, H, W]
        conf_mask: optional [B, T, 1, H, W] or [B, T, H, W]
        """
        B, T, C, H, W = Z.shape
        ref_idx = T // 2
        assert C == self.c
        # enforce divisibility
        assert H % self.p_h == 0 and W % self.p_w == 0, "H/W must be divisible by patch size"
        ph = H // self.p_h
        pw = W // self.p_w
        Np = ph * pw  # #patches per frame
    
        # 1) Patchify per-frame into (B, T, Np, patch_volume)
        z_p = rearrange(Z, 'b t c (hp ph) (wp pw) -> b t (hp wp) (c ph pw)', ph=self.p_h, pw=self.p_w)
        # shape: [B, T, Np, V]
        B, T, Np, V = z_p.shape
        # project patches to tokens
        tokens = self.patch_proj(z_p)  # [B, T, Np, D]

        # optional per-patch confidence
        if conf_mask is not None:
            cm = conf_mask
            if cm.dim() == 5:
                cm = cm.view(B, T, 1, H, W)
                cm = rearrange(cm, 'b t 1 (hp ph) (wp pw) -> b t (hp wp) (ph pw)', ph=self.p_h, pw=self.p_w)
                cm = cm.mean(dim=-1, keepdim=True)  # [B, T, Np, 1]
            else:
                cm = rearrange(cm, 'b t (hp ph) (wp pw) -> b t (hp wp) (ph pw)', ph=self.p_h, pw=self.p_w)
                cm = cm.mean(dim=-1, keepdim=True)
            tokens = tokens * (1.0 + cm)

        # 2) Temporal attention per-patch: for each patch index i, attend across T
        # reshape to (B * Np, T, D) and run temporal blocks
        t_tokens = rearrange(tokens, 'b t n d -> (b n) t d')
        for blk in self.temporal_blocks:
            t_tokens = blk(t_tokens)  # residual inside
        t_tokens = rearrange(t_tokens, '(b n) t d -> b t n d', b=B, n=Np)

        # gating (per token)
        gate_vals = self.gate(t_tokens)  # [B, T, Np, 1]
        t_tokens = t_tokens * gate_vals

        # 3) Spatial windowed attention per-frame:
        # For each frame t, we have tokens [B, Np, D]. Arrange tokens into 2D grid (ph x pw)
        t_out = []
        for tt in range(T):
            frame_tokens = t_tokens[:, tt]  # [B, Np, D]
            frame_grid = rearrange(frame_tokens, 'b (hp wp) d -> b hp wp d', hp=ph, wp=pw)
            # split into windows of win_h x win_w (non-overlapping)
            assert ph % (self.win_h // self.p_h) == 0 and pw % (self.win_w // self.p_w) == 0, \
                "window must align with patch grid"
            # windowize
            ws_h = self.win_h // self.p_h
            ws_w = self.win_w // self.p_w
            windows = rearrange(frame_grid, 'b (Gh gh) (Gw gw) d -> b (Gh Gw) (gh gw) d', gh=ws_h, gw=ws_w)
            # windows: [B, n_windows, tokens_per_window, D]
            Bx, n_win, tok_per_win, D = windows.shape
            windows = windows.view(Bx * n_win, tok_per_win, D)
            # apply spatial transformer blocks (shared)
            for blk in self.spatial_blocks:
                windows = blk(windows)
            # reassemble windows
            windows = windows.view(B, n_win, tok_per_win, D)
            # un-windowize back to frame_grid
            # inverse of 'b (Gh Gw) (gh gw) d -> b (Gh gh) (Gw gw) d'
            Gh = ph // ws_h
            Gw = pw // ws_w
            # For clarity, simply reshape windows back:
            frame_grid = rearrange(windows, 'b (Gh Gw) (gh gw) d -> b (Gh gh) (Gw gw) d', Gh=Gh, Gw=Gw, gh=ws_h, gw=ws_w)
            frame_tokens = rearrange(frame_grid, 'b hp wp d -> b (hp wp) d')
            t_out.append(frame_tokens)
        # t_out: list of length T, each [B, Np, D]
        t_out = torch.stack(t_out, dim=1)  # [B, T, Np, D]

        # 4) Aggregate across time
        temperature = torch.clamp(self.temp_param.abs(), 1e-3, 5.0)
        sigma_t = sigma_t = torch.clamp(self.sigma_param.abs(), 0.5, 10.0)
        energy = t_out.abs().mean(dim=-1, keepdim=True)  # [B, T, Np, 1]
        weights = torch.softmax(energy / temperature, dim=1)
        # reference-centered Gaussian prior
        t_idx = torch.arange(T, device=Z.device).float()
        ref_prior = torch.exp(-((t_idx - ref_idx) ** 2) / (2 * sigma_t ** 2))
        ref_prior = (ref_prior / ref_prior.sum()).view(1, T, 1, 1)  # [1, T, 1, 1]
        weights = weights * ref_prior
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        # print("[~]Debug FusionViT")
        # print(weights.shape)
        # print(t_out.shape)
        fused_tokens = (t_out * weights).sum(dim=1)  # [B, Np, D]

        # 5) Unproject tokens to patch vectors and reconstruct volume residual
        out_patches = self.patch_unproj(fused_tokens)  # [B, Np, V]
        recon = rearrange(out_patches, 'b (hp wp) (c ph pw) -> b c (hp ph) (wp pw)', hp=ph, wp=pw, ph=self.p_h, pw=self.p_w)
        # recon shape: [B, C, H, W]

        z_ref = Z[:, ref_idx]  # [B, C, H, W]
        z_fused = z_ref + self.final_scale * recon
        return z_fused
