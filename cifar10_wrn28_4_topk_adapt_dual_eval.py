# cifar10_wrn28_10_topk_adapt_dual_eval.py
# WideResNet + Top-k activation sparsity (k-Winners) with adaptive keep_ratio control
# Added: optional normalization method (bn / ln / rms)
# Added: configurable activation inside TopKActivation (default: non-sparse LeakyReLU)
# Modified: tighten (compress) happens every N batches (not every epoch)
# Added: dual evaluation
#   - sparse eval: current adaptive keep_ratio
#   - dense  eval: force keep_ratio = 1.0
# Author: ChatGPT (modified per request)

import os
import json
import time
import random
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# =========================
# 0) Utils
# =========================
def set_seed(seed: int = 3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def now_run_dir(root: str = "runs_wrn28_10_topk_adapt"):
    d = os.path.join(root, datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(d, exist_ok=True)
    return d


def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()


# =========================
# 0.5) Optional Norms (bn / ln / rms)
# =========================
class LayerNorm2d(nn.Module):
    """
    Per-pixel LayerNorm for conv feature maps:
    x: (N, C, H, W) -> normalize over C for each (N,H,W) position.
    """
    def __init__(self, num_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_channels = int(num_channels)
        self.eps = float(eps)
        self.affine = bool(affine)

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_perm = x.permute(0, 2, 3, 1)
        y = F.layer_norm(
            x_perm,
            normalized_shape=(self.num_channels,),
            weight=self.weight,
            bias=self.bias,
            eps=self.eps
        )
        return y.permute(0, 3, 1, 2)


class RMSNorm2d(nn.Module):
    """
    Per-pixel RMSNorm for conv feature maps:
    x: (N, C, H, W) -> normalize over C for each (N,H,W) position using RMS (no mean subtraction).
    """
    def __init__(self, num_channels: int, eps: float = 1e-8, affine: bool = True, bias: bool = False):
        super().__init__()
        self.num_channels = int(num_channels)
        self.eps = float(eps)
        self.affine = bool(affine)
        self.use_bias = bool(bias)

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
        else:
            self.register_parameter("weight", None)

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=1, keepdim=True) + self.eps)
        y = x / rms
        if self.weight is not None:
            y = y * self.weight.view(1, -1, 1, 1)
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1)
        return y


def make_norm2d(norm_type: str, num_channels: int, eps: float = 1e-5):
    """
    norm_type:
      - "bn"  : BatchNorm2d
      - "ln"  : LayerNorm2d (per-pixel on channels)
      - "rms" : RMSNorm2d   (per-pixel on channels)
      - "none": Identity
    """
    norm_type = str(norm_type).lower()
    if norm_type == "bn":
        return nn.BatchNorm2d(num_channels, eps=eps, momentum=0.1, affine=True, track_running_stats=True)
    if norm_type == "ln":
        return LayerNorm2d(num_channels, eps=eps, affine=True)
    if norm_type == "rms":
        return RMSNorm2d(num_channels, eps=max(eps, 1e-8), affine=True, bias=True)
    if norm_type == "none":
        return nn.Identity()
    raise ValueError(f"Unknown norm_type={norm_type}. Use one of: bn / ln / rms / none")


# =========================
# 1) Top-k Activation (activation is configurable now)
# =========================
class TopKActivation(nn.Module):
    """
    Apply an activation (configurable) then Hard Top-k sparsification (k-Winners).
    For conv feature maps (N,C,H,W), default mode keeps Top-k channels at each (n,h,w).
    """
    def __init__(
        self,
        keep_ratio: float = 0.2,
        mode: str = "channel",     # "channel" or "global"
        use_abs: bool = False,
        act_type: str = "leakyrelu",
        leaky_slope: float = 0.1,
        act_eps: float = 0.0,
        softplus_beta: float = 1.0,
        softplus_threshold: float = 20.0,
    ):
        super().__init__()
        assert 0.0 <= keep_ratio <= 1.0
        assert mode in ["channel", "global"]
        self.keep_ratio = float(keep_ratio)
        self.mode = mode
        self.use_abs = bool(use_abs)

        self.act_type = str(act_type).lower()
        self.leaky_slope = float(leaky_slope)
        self.act_eps = float(act_eps)
        self.softplus_beta = float(softplus_beta)
        self.softplus_threshold = float(softplus_threshold)

        self.last_nz_ratio = None

    @torch.no_grad()
    def _compute_nz_ratio(self, y: torch.Tensor) -> float:
        return float((y != 0).float().mean().item())

    def _activate(self, x: torch.Tensor) -> torch.Tensor:
        t = self.act_type
        if t == "relu":
            x = F.relu(x, inplace=False)
        elif t in ["lrelu", "leakyrelu"]:
            x = F.leaky_relu(x, negative_slope=self.leaky_slope, inplace=False)
        elif t == "elu":
            x = F.elu(x, inplace=False)
        elif t == "gelu":
            x = F.gelu(x)
        elif t in ["silu", "swish"]:
            x = F.silu(x)
        elif t == "softplus":
            x = F.softplus(x, beta=self.softplus_beta, threshold=self.softplus_threshold)
        elif t in ["id", "identity", "none"]:
            x = x
        else:
            raise ValueError(f"Unknown act_type={self.act_type}. Use: relu/leakyrelu/elu/gelu/silu/softplus/identity")

        if self.act_eps != 0.0:
            x = x + self.act_eps
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._activate(x)

        if self.keep_ratio >= 1.0 - 1e-12:
            self.last_nz_ratio = self._compute_nz_ratio(x)
            return x

        if x.dim() == 2:
            dim = 1
            D = x.size(dim)
            k = max(1, int(D * self.keep_ratio))
            score = x.abs() if self.use_abs else x
            _, idx = torch.topk(score, k=k, dim=dim, largest=True, sorted=False)
            mask = torch.zeros_like(x)
            mask.scatter_(dim, idx, 1.0)
            y = x * mask
            self.last_nz_ratio = self._compute_nz_ratio(y)
            return y

        if x.dim() == 4:
            N, C, H, W = x.shape
            score = x.abs() if self.use_abs else x

            if self.mode == "channel":
                k = max(1, int(C * self.keep_ratio))
                _, idx = torch.topk(score, k=k, dim=1, largest=True, sorted=False)  # (N,k,H,W)
                mask = torch.zeros_like(x)
                mask.scatter_(1, idx, 1.0)
                y = x * mask
                self.last_nz_ratio = self._compute_nz_ratio(y)
                return y

            if self.mode == "global":
                D = C * H * W
                k = max(1, int(D * self.keep_ratio))
                flat = score.reshape(N, -1)
                _, idx = torch.topk(flat, k=k, dim=1, largest=True, sorted=False)
                mask = torch.zeros_like(flat)
                mask.scatter_(1, idx, 1.0)
                mask = mask.view(N, C, H, W)
                y = x * mask
                self.last_nz_ratio = self._compute_nz_ratio(y)
                return y

        dim = -1
        D = x.size(dim)
        k = max(1, int(D * self.keep_ratio))
        score = x.abs() if self.use_abs else x
        _, idx = torch.topk(score, k=k, dim=dim, largest=True, sorted=False)
        mask = torch.zeros_like(x)
        mask.scatter_(dim, idx, 1.0)
        y = x * mask
        self.last_nz_ratio = self._compute_nz_ratio(y)
        return y


@torch.no_grad()
def collect_topk_nz_stats(model: nn.Module):
    ratios = []
    for m in model.modules():
        if isinstance(m, TopKActivation) and (m.last_nz_ratio is not None):
            ratios.append(m.last_nz_ratio)
    if len(ratios) == 0:
        return None
    return float(np.mean(ratios))


@torch.no_grad()
def override_all_topk_keep_ratio(model: nn.Module, keep_ratio: float):
    topk_modules = []
    old_keep_ratios = []
    for m in model.modules():
        if isinstance(m, TopKActivation):
            topk_modules.append(m)
            old_keep_ratios.append(float(m.keep_ratio))
            m.keep_ratio = float(keep_ratio)
    return topk_modules, old_keep_ratios


@torch.no_grad()
def restore_all_topk_keep_ratio(topk_modules, old_keep_ratios):
    for m, old_k in zip(topk_modules, old_keep_ratios):
        m.keep_ratio = float(old_k)


# =========================
# 2) Adaptive keep_ratio controller (tighten every N batches)
# =========================
class AdaptiveKeepRatioController:
    """
    每个 batch 更新 EMA，但只有每 N 个 batch 才“决策”一次：
      - 如果 EMA 相对 best_ema 下降超过 drop_tol -> relax (增加 keep)
      - 否则 -> tighten：keep <- keep*(1-tighten_pct)
    cooldown_epochs：这里表示“tighten 决策点”的冷却次数（不是 epoch）
    """
    def __init__(
        self,
        keep_init: float = 1.0,
        keep_min: float = 0.2,
        keep_max: float = 1.0,
        tighten_pct: float = 0.02,
        relax_step: float = 0.03,
        ema_beta: float = 0.9,
        drop_tol: float = 0.01,
        cooldown_epochs: int = 3,
        relax_gain: float = 1.5,
        max_relax_jump: float = 0.03,
        tighten_every_n_batches: int = 50,
    ):
        self.keep_min = float(keep_min)
        self.keep_max = float(keep_max)
        self.keep = float(np.clip(keep_init, self.keep_min, self.keep_max))

        self.tighten_pct = float(tighten_pct)
        assert 0.0 <= self.tighten_pct < 1.0, "tighten_pct should be in [0, 1)."

        self.relax_step = float(relax_step)
        self.ema_beta = float(ema_beta)
        self.drop_tol = float(drop_tol)

        self.cooldown_epochs = int(cooldown_epochs)
        self.relax_gain = float(relax_gain)
        self.max_relax_jump = float(max_relax_jump)

        self.tighten_every_n_batches = int(max(1, tighten_every_n_batches))
        self.batch_step = 0

        self.ema_acc = None
        self.best_ema = None
        self.cooldown_left = 0

    def apply_to(self, model: nn.Module):
        for m in model.modules():
            if isinstance(m, TopKActivation):
                m.keep_ratio = float(self.keep)

    def step_batch(self, train_acc_batch: float) -> bool:
        """
        Called every batch.
        Returns True if keep_ratio changed at this step (so caller can re-apply to model).
        """
        self.batch_step += 1

        if self.ema_acc is None:
            self.ema_acc = float(train_acc_batch)
            self.best_ema = float(train_acc_batch)
        else:
            self.ema_acc = self.ema_beta * self.ema_acc + (1.0 - self.ema_beta) * float(train_acc_batch)
            if self.ema_acc > self.best_ema:
                self.best_ema = self.ema_acc

        if (self.batch_step % self.tighten_every_n_batches) != 0:
            return False

        drop = self.best_ema - self.ema_acc

        # Case 1) drop => relax
        if drop > self.drop_tol:
            scale = 1.0 + self.relax_gain * ((drop - self.drop_tol) / max(self.drop_tol, 1e-8))
            delta = self.relax_step * min(scale, 5.0)
            if self.max_relax_jump > 0:
                delta = min(delta, self.max_relax_jump)
            new_keep = min(self.keep_max, self.keep + delta)
            changed = (abs(new_keep - self.keep) > 1e-12)
            self.keep = new_keep
            self.cooldown_left = self.cooldown_epochs
            return changed

        # Case 2) no drop => tighten (with cooldown)
        if self.cooldown_left > 0:
            self.cooldown_left -= 1
            return False

        new_keep = max(self.keep_min, self.keep * (1.0 - self.tighten_pct))
        changed = (abs(new_keep - self.keep) > 1e-12)
        self.keep = new_keep
        return changed


# =========================
# 3) WideResNet (WRN-28-k)
# =========================
def conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)


class WideBasicBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int,
        drop_rate: float,
        topk_keep_ratio: float,
        topk_mode: str = "channel",
        topk_use_abs: bool = False,
        act_type: str = "leakyrelu",
        act_leaky_slope: float = 0.1,
        act_eps: float = 0.0,
        norm_type: str = "bn",
        norm_eps: float = 1e-5,
    ):
        super().__init__()

        self.norm1 = make_norm2d(norm_type, in_ch, eps=norm_eps)
        self.act1 = TopKActivation(
            keep_ratio=topk_keep_ratio, mode=topk_mode, use_abs=topk_use_abs,
            act_type=act_type, leaky_slope=act_leaky_slope, act_eps=act_eps
        )
        self.conv1 = conv3x3(in_ch, out_ch, stride=stride)

        self.norm2 = make_norm2d(norm_type, out_ch, eps=norm_eps)
        self.act2 = TopKActivation(
            keep_ratio=topk_keep_ratio, mode=topk_mode, use_abs=topk_use_abs,
            act_type=act_type, leaky_slope=act_leaky_slope, act_eps=act_eps
        )
        self.conv2 = conv3x3(out_ch, out_ch, stride=1)

        self.drop_rate = float(drop_rate)
        self.shortcut = None
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = self.conv1(self.act1(self.norm1(x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(self.act2(self.norm2(out)))

        if self.shortcut is not None:
            x = self.shortcut(x)
        return out + x


class WideResNet(nn.Module):
    """
    WRN depth = 6n + 4. For WRN-28, n=4.
    """
    def __init__(
        self,
        depth: int = 28,
        widen_factor: int = 10,
        num_classes: int = 10,
        drop_rate: float = 0.0,
        topk_keep_ratio: float = 1.0,
        topk_mode: str = "channel",
        topk_use_abs: bool = False,
        act_type: str = "leakyrelu",
        act_leaky_slope: float = 0.1,
        act_eps: float = 0.0,
        norm_type: str = "bn",
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        assert (depth - 4) % 6 == 0, "WRN depth should be 6n+4"
        n = (depth - 4) // 6
        k = widen_factor

        stages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = conv3x3(3, stages[0], stride=1)

        self.block1 = self._make_group(
            stages[0], stages[1], n, stride=1, drop_rate=drop_rate,
            topk_keep_ratio=topk_keep_ratio, topk_mode=topk_mode, topk_use_abs=topk_use_abs,
            act_type=act_type, act_leaky_slope=act_leaky_slope, act_eps=act_eps,
            norm_type=norm_type, norm_eps=norm_eps
        )
        self.block2 = self._make_group(
            stages[1], stages[2], n, stride=2, drop_rate=drop_rate,
            topk_keep_ratio=topk_keep_ratio, topk_mode=topk_mode, topk_use_abs=topk_use_abs,
            act_type=act_type, act_leaky_slope=act_leaky_slope, act_eps=act_eps,
            norm_type=norm_type, norm_eps=norm_eps
        )
        self.block3 = self._make_group(
            stages[2], stages[3], n, stride=2, drop_rate=drop_rate,
            topk_keep_ratio=topk_keep_ratio, topk_mode=topk_mode, topk_use_abs=topk_use_abs,
            act_type=act_type, act_leaky_slope=act_leaky_slope, act_eps=act_eps,
            norm_type=norm_type, norm_eps=norm_eps
        )

        self.norm = make_norm2d(norm_type, stages[3], eps=norm_eps)
        self.act = TopKActivation(
            keep_ratio=topk_keep_ratio, mode=topk_mode, use_abs=topk_use_abs,
            act_type=act_type, leaky_slope=act_leaky_slope, act_eps=act_eps
        )
        self.fc = nn.Linear(stages[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                nn.init.zeros_(m.bias)

    def _make_group(
        self, in_ch, out_ch, n_blocks, stride, drop_rate,
        topk_keep_ratio, topk_mode, topk_use_abs,
        act_type, act_leaky_slope, act_eps,
        norm_type, norm_eps
    ):
        layers = []
        for i in range(n_blocks):
            s = stride if i == 0 else 1
            ch_in = in_ch if i == 0 else out_ch
            layers.append(WideBasicBlock(
                ch_in, out_ch, s, drop_rate=drop_rate,
                topk_keep_ratio=topk_keep_ratio, topk_mode=topk_mode, topk_use_abs=topk_use_abs,
                act_type=act_type, act_leaky_slope=act_leaky_slope, act_eps=act_eps,
                norm_type=norm_type, norm_eps=norm_eps
            ))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.act(self.norm(out))
        out = F.adaptive_avg_pool2d(out, 1).view(out.size(0), -1)
        return self.fc(out)


# =========================
# 4) Training Loop
# =========================
@dataclass
class CFG:
    # data
    batch_size: int = 128
    num_workers: int = 4

    # model
    depth: int = 28
    widen_factor: int = 4
    drop_rate: float = 0.0

    # norm
    norm_type: str = "rms"     # "bn" | "ln" | "rms" | "none"
    norm_eps: float = 1e-10

    # activation inside TopKActivation
    act_type: str = "relu"
    act_leaky_slope: float = 0.1
    act_eps: float = 0.0

    # Top-k mode
    topk_mode: str = "global"   # "channel" or "global"
    topk_use_abs: bool = False

    # Adaptive keep_ratio control
    keep_init: float = 1.0
    keep_min: float = 0.0
    keep_max: float = 1.0
    tighten_pct: float = 0.018
    relax_step: float = 1.0
    ema_beta: float = 0.5
    drop_tol: float = 0.12
    cooldown_epochs: int = 3
    relax_gain: float = 1.5
    max_relax_jump: float = 1.0

    # tighten every N batches
    tighten_every_n_batches: int = 128

    # eval
    eval_dense_keep_ratio: float = 1.0

    # optim
    epochs: int = 500
    lr: float = 0.1
    lr_ref_bs: int = 128
    lr_scale_with_bs: bool = True
    momentum: float = 0.9
    weight_decay: float = 0.0
    nesterov: bool = True

    # sched
    cosine: bool = True

    # misc
    seed: int = 3407
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True
    log_interval: int = 100

    def __post_init__(self):
        if self.lr_scale_with_bs and self.lr_ref_bs > 0:
            self.lr = float(self.lr) * float(self.batch_size) / float(self.lr_ref_bs)


def build_dataloaders(cfg: CFG):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)

    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True
    )
    test_loader = DataLoader(
        test_set, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )
    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, scaler, device, cfg: CFG, keep_ctrl: AdaptiveKeepRatioController):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    keep_ctrl.apply_to(model)

    t0 = time.time()
    for it, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if cfg.amp and device.startswith("cuda"):
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()

        bs = x.size(0)
        batch_acc = accuracy_top1(logits.detach(), y)

        total_loss += loss.item() * bs
        total_acc += batch_acc * bs
        n += bs

        changed = keep_ctrl.step_batch(batch_acc)
        if changed:
            keep_ctrl.apply_to(model)

        if (it + 1) % cfg.log_interval == 0:
            dt = time.time() - t0
            avg_loss = total_loss / n
            avg_acc = total_acc / n
            nz = collect_topk_nz_stats(model)
            print(
                f"  iter {it+1:4d}/{len(loader)} | loss {avg_loss:.4f} | acc {avg_acc*100:.2f}%"
                + (f" | nz {nz*100:.2f}%" if nz is not None else "")
                + f" | keep {keep_ctrl.keep:.3f} | EMA {keep_ctrl.ema_acc*100:.2f}% | {dt:.1f}s"
            )
            t0 = time.time()

    avg_loss = total_loss / n
    avg_acc = total_acc / n
    avg_nz = collect_topk_nz_stats(model)
    return avg_loss, avg_acc, avg_nz


@torch.no_grad()
def evaluate(model, loader, device, cfg: CFG, force_keep_ratio=None):
    """
    force_keep_ratio:
      - None: use current model keep_ratio (sparse eval)
      - 1.0 : force dense eval
      - other float in [0,1]: force a specific eval keep_ratio
    """
    model.eval()

    topk_modules = None
    old_keep_ratios = None

    if force_keep_ratio is not None:
        topk_modules, old_keep_ratios = override_all_topk_keep_ratio(model, force_keep_ratio)

    total_loss = 0.0
    total_acc = 0.0
    n = 0

    try:
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            loss = F.cross_entropy(logits, y)

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_acc += accuracy_top1(logits, y) * bs
            n += bs

        avg_loss = total_loss / n
        avg_acc = total_acc / n
        avg_nz = collect_topk_nz_stats(model)
        return avg_loss, avg_acc, avg_nz

    finally:
        if force_keep_ratio is not None:
            restore_all_topk_keep_ratio(topk_modules, old_keep_ratios)


# =========================
# 5) Plotting
# =========================
class LivePlotter:
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        self.fig = plt.figure(figsize=(12, 4))
        self.ax1 = self.fig.add_subplot(1, 2, 1)
        self.ax2 = self.fig.add_subplot(1, 2, 2)

        self.epochs = []

        self.train_loss = []
        self.test_loss_sparse = []
        self.test_loss_dense = []

        self.train_acc = []
        self.test_acc_sparse = []
        self.test_acc_dense = []

        self.nz_train = []
        self.nz_test_sparse = []
        self.nz_test_dense = []

        self.keep_ratio = []
        self.ema_acc = []

        plt.ion()
        plt.show()

    def update(
        self,
        epoch,
        tr_loss,
        te_loss_sparse,
        te_loss_dense,
        tr_acc,
        te_acc_sparse,
        te_acc_dense,
        nz_tr,
        nz_te_sparse,
        nz_te_dense,
        keep_ratio,
        ema_acc
    ):
        self.epochs.append(epoch)

        self.train_loss.append(tr_loss)
        self.test_loss_sparse.append(te_loss_sparse)
        self.test_loss_dense.append(te_loss_dense)

        self.train_acc.append(tr_acc)
        self.test_acc_sparse.append(te_acc_sparse)
        self.test_acc_dense.append(te_acc_dense)

        self.nz_train.append(nz_tr if nz_tr is not None else np.nan)
        self.nz_test_sparse.append(nz_te_sparse if nz_te_sparse is not None else np.nan)
        self.nz_test_dense.append(nz_te_dense if nz_te_dense is not None else np.nan)

        self.keep_ratio.append(keep_ratio)
        self.ema_acc.append(ema_acc)

        self.ax1.clear()
        self.ax2.clear()

        self.ax1.plot(self.epochs, self.train_loss, label="train loss")
        self.ax1.plot(self.epochs, self.test_loss_sparse, label="test sparse loss")
        self.ax1.plot(self.epochs, self.test_loss_dense, label="test dense loss", linestyle="--")
        self.ax1.set_title("Loss")
        self.ax1.legend()

        self.ax2.plot(self.epochs, np.array(self.train_acc) * 100.0, label="train acc (%)")
        self.ax2.plot(self.epochs, np.array(self.test_acc_sparse) * 100.0, label="test sparse acc (%)")
        self.ax2.plot(self.epochs, np.array(self.test_acc_dense) * 100.0, label="test dense acc (%)", linestyle="--")
        self.ax2.plot(self.epochs, np.array(self.ema_acc) * 100.0, label="train EMA acc (%)", linestyle="-.")
        self.ax2.plot(self.epochs, np.array(self.nz_train) * 100.0, label="nz train (%)")
        self.ax2.plot(self.epochs, np.array(self.nz_test_sparse) * 100.0, label="nz test sparse (%)")
        self.ax2.plot(self.epochs, np.array(self.nz_test_dense) * 100.0, label="nz test dense (%)", linestyle="--")
        self.ax2.plot(self.epochs, np.array(self.keep_ratio) * 100.0, label="keep_ratio (%)", linestyle=":")
        self.ax2.set_title("Acc / Sparsity / Keep")
        self.ax2.legend()

        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.fig.savefig(os.path.join(self.save_dir, "curves.png"), dpi=180)


# =========================
# 6) Main
# =========================
def main():
    cfg = CFG()
    set_seed(cfg.seed)

    torch.backends.cudnn.benchmark = True

    run_dir = now_run_dir("runs_wrn28_10_topk_adapt_dual_eval")
    print("Run dir:", run_dir)

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    device = cfg.device
    print("Device:", device)
    print(f"Norm type: {cfg.norm_type} (eps={cfg.norm_eps})")
    print(f"Act type : {cfg.act_type} (leaky_slope={cfg.act_leaky_slope}, act_eps={cfg.act_eps})")
    print(f"TopK     : mode={cfg.topk_mode}, use_abs={cfg.topk_use_abs}")
    print(f"Tighten  : every {cfg.tighten_every_n_batches} batches")
    print(f"Dense eval keep_ratio = {cfg.eval_dense_keep_ratio:.3f}")

    train_loader, test_loader = build_dataloaders(cfg)

    model = WideResNet(
        depth=cfg.depth,
        widen_factor=cfg.widen_factor,
        num_classes=10,
        drop_rate=cfg.drop_rate,
        topk_keep_ratio=cfg.keep_init,
        topk_mode=cfg.topk_mode,
        topk_use_abs=cfg.topk_use_abs,
        act_type=cfg.act_type,
        act_leaky_slope=cfg.act_leaky_slope,
        act_eps=cfg.act_eps,
        norm_type=cfg.norm_type,
        norm_eps=cfg.norm_eps,
    ).to(device)

    keep_ctrl = AdaptiveKeepRatioController(
        keep_init=cfg.keep_init,
        keep_min=cfg.keep_min,
        keep_max=cfg.keep_max,
        tighten_pct=cfg.tighten_pct,
        relax_step=cfg.relax_step,
        ema_beta=cfg.ema_beta,
        drop_tol=cfg.drop_tol,
        cooldown_epochs=cfg.cooldown_epochs,
        relax_gain=cfg.relax_gain,
        max_relax_jump=cfg.max_relax_jump,
        tighten_every_n_batches=cfg.tighten_every_n_batches,
    )

    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        nesterov=cfg.nesterov
    )

    if cfg.cosine:
        lr_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    else:
        lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.startswith("cuda")))
    plotter = LivePlotter(run_dir)

    best_acc_dense = 0.0
    best_acc_sparse = 0.0
    best_path_prev = None
    log_path = os.path.join(run_dir, "log.jsonl")

    for epoch in range(1, cfg.epochs + 1):
        keep_ctrl.apply_to(model)

        lr_now = float(optimizer.param_groups[0]["lr"])
        print(
            f"\nEpoch {epoch:03d}/{cfg.epochs} | lr={lr_now:.5f} | keep_ratio={keep_ctrl.keep:.3f} "
            f"| EMA_acc={(keep_ctrl.ema_acc*100 if keep_ctrl.ema_acc is not None else float('nan')):.2f}% "
            f"| cooldown={keep_ctrl.cooldown_left} | step={keep_ctrl.batch_step}"
        )

        # train
        tr_loss, tr_acc, tr_nz = train_one_epoch(model, train_loader, optimizer, scaler, device, cfg, keep_ctrl)

        # eval 1: sparse (current keep_ratio)
        te_loss_sparse, te_acc_sparse, te_nz_sparse = evaluate(
            model, test_loader, device, cfg, force_keep_ratio=None
        )

        # eval 2: dense (force keep_ratio = 1.0)
        te_loss_dense, te_acc_dense, te_nz_dense = evaluate(
            model, test_loader, device, cfg, force_keep_ratio=cfg.eval_dense_keep_ratio
        )

        lr_sched.step()

        best_acc_sparse = max(best_acc_sparse, te_acc_sparse)

        rec = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),

            "keep_ratio_train": float(keep_ctrl.keep),
            "keep_ratio_used_sparse_eval": float(model.act.keep_ratio),
            "keep_ratio_used_dense_eval": float(cfg.eval_dense_keep_ratio),

            "train_loss": float(tr_loss),
            "train_acc": float(tr_acc),
            "train_nz": None if tr_nz is None else float(tr_nz),

            "test_sparse_loss": float(te_loss_sparse),
            "test_sparse_acc": float(te_acc_sparse),
            "test_sparse_nz": None if te_nz_sparse is None else float(te_nz_sparse),

            "test_dense_loss": float(te_loss_dense),
            "test_dense_acc": float(te_acc_dense),
            "test_dense_nz": None if te_nz_dense is None else float(te_nz_dense),

            "ema_train_acc": None if keep_ctrl.ema_acc is None else float(keep_ctrl.ema_acc),
            "best_ema_train_acc": None if keep_ctrl.best_ema is None else float(keep_ctrl.best_ema),
            "best_test_sparse_acc_so_far": float(best_acc_sparse),
            "best_test_dense_acc_so_far": float(best_acc_dense),

            "cooldown_left": int(keep_ctrl.cooldown_left),
            "batch_step": int(keep_ctrl.batch_step),
            "tighten_every_n_batches": int(cfg.tighten_every_n_batches),

            "time": datetime.now().isoformat(timespec="seconds"),
            "norm_type": cfg.norm_type,
            "norm_eps": cfg.norm_eps,
            "act_type": cfg.act_type,
            "topk_mode": cfg.topk_mode,
            "topk_use_abs": cfg.topk_use_abs,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        plotter.update(
            epoch=epoch,
            tr_loss=tr_loss,
            te_loss_sparse=te_loss_sparse,
            te_loss_dense=te_loss_dense,
            tr_acc=tr_acc,
            te_acc_sparse=te_acc_sparse,
            te_acc_dense=te_acc_dense,
            nz_tr=tr_nz,
            nz_te_sparse=te_nz_sparse,
            nz_te_dense=te_nz_dense,
            keep_ratio=float(model.act.keep_ratio),
            ema_acc=float(keep_ctrl.ema_acc) if keep_ctrl.ema_acc is not None else np.nan
        )

        print(f"  train       : loss={tr_loss:.4f}, acc={tr_acc*100:.2f}%, nz={(tr_nz*100 if tr_nz is not None else float('nan')):.2f}%")
        print(f"  test_sparse : loss={te_loss_sparse:.4f}, acc={te_acc_sparse*100:.2f}%, nz={(te_nz_sparse*100 if te_nz_sparse is not None else float('nan')):.2f}%")
        print(f"  test_dense  : loss={te_loss_dense:.4f}, acc={te_acc_dense*100:.2f}%, nz={(te_nz_dense*100 if te_nz_dense is not None else float('nan')):.2f}%")
        print(f"  keep        : now={keep_ctrl.keep:.3f} | EMA={keep_ctrl.ema_acc*100:.2f}% | bestEMA={keep_ctrl.best_ema*100:.2f}%")
        print(f"  best so far : sparse={best_acc_sparse*100:.2f}% | dense={best_acc_dense*100:.2f}%")

        # 默认按 dense acc 保存 best
        if te_acc_dense > best_acc_dense:
            best_acc_dense = te_acc_dense

            if best_path_prev is not None and os.path.exists(best_path_prev):
                try:
                    os.remove(best_path_prev)
                except OSError:
                    pass

            best_name = f"best_dense_te{best_acc_dense*100:.2f}_e{epoch:04d}.pt"
            best_path = os.path.join(run_dir, best_name)

            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "best_acc_dense": best_acc_dense,
                    "best_acc_sparse_so_far": best_acc_sparse,
                    "cfg": asdict(cfg),
                },
                best_path
            )
            best_path_prev = best_path
            print(f"  ✅ best(dense) updated: {best_acc_dense*100:.2f}% -> {best_name}")

        if epoch % 50 == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "cfg": asdict(cfg),
                },
                os.path.join(run_dir, f"ckpt_e{epoch:03d}.pt")
            )

    torch.save(
        {
            "model": model.state_dict(),
            "epoch": cfg.epochs,
            "best_acc_dense": best_acc_dense,
            "best_acc_sparse": best_acc_sparse,
            "cfg": asdict(cfg),
        },
        os.path.join(run_dir, "last.pt")
    )

    print(f"\nDone. best_dense_acc = {best_acc_dense*100:.2f}% | best_sparse_acc = {best_acc_sparse*100:.2f}%")

    plt.ioff()
    plt.show(block=True)


if __name__ == "__main__":
    main()