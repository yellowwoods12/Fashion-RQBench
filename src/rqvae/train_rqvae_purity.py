#!/usr/bin/env python3
"""
RQ-VAE training with an exclusivity auxiliary loss.

Cluster-purity comparisons showed vanilla reconstruction+VQ already gets most
single attributes to >=90% cluster purity on its own — no auxiliary purity
head needed. The one real gap is exclusivity: the same attribute *combination*
ending up scattered across multiple clusters instead of owned by one.

  L_exclusivity — supervised contrastive loss (SupCon) on each level's own
                  quantized representation. Items sharing the same attribute
                  combination are pulled together; items with different
                  combinations are pushed apart → exclusivity / between-
                  cluster separation. The finest level pulls on *all*
                  articleType-anchored attribute combinations at once.

Total loss:
  L = L_recon + L_vq + λ_excl * L_exclusivity

Attribute columns used as supervision labels (match RQ-VAE item_context):
  masterCategory, articleType, gender, baseColour, usage, season, ...
  (extend via ATTR_COLS / _COMBO_COLS)
"""

import inspect
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

import wandb
from device_manager import DeviceManager
from logger import setup_logger

logger = setup_logger("train-rqvae-purity", log_to_file=True)

# Attribute columns used as SupCon supervision labels (directly or via combos).
# Columns directly in the parquet are used as-is.
# "price_tier" is derived from "discountedPrice".
ATTR_COLS = [
    # High-level taxonomy (directly in parquet, 0 nulls)
    "masterCategory",   # 7 classes
    "articleType",      # 143 classes
    "gender",           # 5 classes
    "ageGroup",         # low cardinality
    # Visual attributes
    "baseColour",       # 47 classes
    "season",           # 5 classes
    # Usage / style
    "usage",            # 10 classes
    "fashionType",      # 3 classes
    # Price segment
    "price_tier",       # budget / mid-range / premium  (derived)
    # Combo labels — 100% filled (both parents are 100% filled), directly encode
    # the joint attribute signal that matters most for benchmark query specificity.
    # All anchored on articleType — a shopping query always implies a product type.
    "combo_articletype_colour",       # articleType + "__" + baseColour
    "combo_articletype_gender",       # articleType + "__" + gender
    "combo_articletype_usage",        # articleType + "__" + usage
    "combo_articletype_season",       # articleType + "__" + season
    "combo_articletype_agegroup",     # articleType + "__" + ageGroup
    "combo_articletype_pricetier",    # articleType + "__" + price_tier
    "combo_articletype_fashiontype",  # articleType + "__" + fashionType
]

# Which parquet columns are combined to form each combo label
_COMBO_COLS = {
    "combo_articletype_colour":      ("articleType", "baseColour"),
    "combo_articletype_gender":      ("articleType", "gender"),
    "combo_articletype_usage":       ("articleType", "usage"),
    "combo_articletype_season":      ("articleType", "season"),
    "combo_articletype_agegroup":    ("articleType", "ageGroup"),
    "combo_articletype_pricetier":   ("articleType", "price_tier"),
    "combo_articletype_fashiontype": ("articleType", "fashionType"),
}

# Per-level SupCon labels: each level gets a progressively finer grouping signal.
# L0 = masterCategory (coarse, 7 classes)   → forces the 64 root buckets to separate top categories
# L1 = articleType (143 classes)            → forces mid-level buckets to separate product types
# L2 = all articleType x <attr> combos      → forces fine buckets to be exclusive for every
#                                              attribute combination at once, not just colour
SUPCON_LABELS_PER_LEVEL = [
    ["masterCategory"],
    ["articleType"],
    [
        "combo_articletype_colour",
        "combo_articletype_gender",
        "combo_articletype_usage",
        "combo_articletype_season",
        "combo_articletype_agegroup",
        "combo_articletype_pricetier",
        "combo_articletype_fashiontype",
    ],
]


@dataclass
class RQVAEPurityConfig:
    """Configuration for RQ-VAE training with an exclusivity loss."""

    # Data
    category: str = "public_dataset"
    data_dir: Path = field(default_factory=lambda: Path("data"))
    embeddings_path: Optional[Path] = None
    checkpoint_dir: Path = field(
        default_factory=lambda: Path("checkpoints") / "rqvae_purity_v3_public_dataset"
    )

    # Model
    item_embedding_dim: int = 1024
    encoder_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    codebook_embedding_dim: int = 32
    codebook_quantization_levels: int = 3
    codebook_size: int = 64
    commitment_weight: float = 0.25
    use_rotation_trick: bool = True

    # Exclusivity loss — per-level SupCon, each level gets a list of grouping
    # labels (summed/averaged if a level has more than one). Outer list length
    # must match codebook_quantization_levels. Matches SUPCON_LABELS_PER_LEVEL
    # defined above.
    supcon_labels_per_level: List[List[str]] = field(
        default_factory=lambda: [list(labels) for labels in SUPCON_LABELS_PER_LEVEL]
    )
    supcon_temperature: float = 0.07
    lambda_exclusivity: float = 0.05

    # Training
    batch_size: int = 512
    gradient_accumulation_steps: int = 4
    num_epochs: int = 50000
    scheduler_type: str = "cosine_with_warmup"
    warmup_start_lr: float = 1e-8
    warmup_steps: int = 200
    max_lr: float = 3e-4
    min_lr: float = 1e-6
    use_gradient_clipping: bool = True
    gradient_clip_norm: float = 1.0
    use_kmeans_init: bool = True
    reset_unused_codes: bool = True
    steps_per_codebook_reset: int = 2
    codebook_usage_threshold: float = 1.0
    val_split: float = 0.05

    # Logging
    steps_per_train_log: int = 1000
    steps_per_val_log: int = 2000

    def __post_init__(self):
        if self.embeddings_path is None:
            self.embeddings_path = (
                self.data_dir / "output" / f"{self.category}_items_with_embeddings_1024dim.parquet"
            )
        assert len(self.supcon_labels_per_level) == self.codebook_quantization_levels, (
            f"supcon_labels_per_level has {len(self.supcon_labels_per_level)} entries, "
            f"expected {self.codebook_quantization_levels} (one per quantization level)"
        )


# ── Dataset ───────────────────────────────────────────────────────────────────

def _price_tier(price) -> Optional[str]:
    try:
        p = float(price)
    except (TypeError, ValueError):
        return None
    if p < 711:
        return "budget"
    if p <= 1595:
        return "mid-range"
    return "premium"


class EmbeddingDatasetWithLabels(Dataset):
    """
    Loads embeddings + integer attribute labels for exclusivity (SupCon) supervision.

    Handles two kinds of columns:
      - Direct parquet columns (masterCategory, gender, …)
      - price_tier derived from "discountedPrice"
    Combo columns (e.g. combo_articletype_colour) are built from their parent
    columns, which must therefore also be listed in attr_cols.
    Missing values are encoded as -1 and masked out in all loss functions.
    """

    def __init__(self, embeddings_path: str, attr_cols: List[str], limit: Optional[int] = None):
        for combo_col, (col_a, col_b) in _COMBO_COLS.items():
            if combo_col in attr_cols:
                missing = [c for c in (col_a, col_b) if c not in attr_cols]
                assert not missing, f"'{combo_col}' needs parent column(s) {missing} in attr_cols"

        logger.info(f"Loading embeddings + labels from {embeddings_path}")
        df = pl.read_parquet(embeddings_path)
        if limit is not None:
            df = df.head(limit)

        self.embeddings = torch.tensor(df["embedding"].to_list(), dtype=torch.float32)
        self.item_ids = df["id"].to_list()

        # Derive price_tier
        price_tiers: List[Optional[str]] = []
        if "price_tier" in attr_cols and "discountedPrice" in df.columns:
            price_tiers = [_price_tier(p) for p in df["discountedPrice"].to_list()]
            logger.info(f"  Derived price_tier: {len(set(t for t in price_tiers if t))} tiers")

        # Build a column-name → list-of-values mapping
        col_values: Dict[str, list] = {}
        for col in attr_cols:
            if col == "price_tier":
                if price_tiers:
                    col_values[col] = price_tiers
            elif col in _COMBO_COLS:
                continue  # built from parent columns below, not loaded directly
            elif col in df.columns:
                col_values[col] = df[col].to_list()
            else:
                logger.warning(f"  Column '{col}' not found in parquet — skipping")

        # Build combo columns from already-loaded base columns
        n_items = len(self.embeddings)
        for combo_col, (col_a, col_b) in _COMBO_COLS.items():
            if combo_col not in attr_cols:
                continue
            vals_a = col_values.get(col_a, [None] * n_items)
            vals_b = col_values.get(col_b, [None] * n_items)
            col_values[combo_col] = [
                f"{a}__{b}" if a is not None and b is not None else None
                for a, b in zip(vals_a, vals_b)
            ]
            logger.info(f"  Built combo column '{combo_col}' from {col_a} x {col_b}")

        # Encode each column to integer labels; -1 for nulls / missing
        self.label_tensors: Dict[str, Tensor] = {}
        self.num_classes: Dict[str, int] = {}
        self.fill_rates: Dict[str, float] = {}
        self.encoders: Dict[str, LabelEncoder] = {}

        for col, values in col_values.items():
            non_null = [v for v in values if v is not None]
            if len(non_null) < 2:
                logger.warning(f"  '{col}' has < 2 non-null values — skipping")
                continue
            le = LabelEncoder()
            le.fit(non_null)
            encoded = [int(le.transform([v])[0]) if v is not None else -1 for v in values]
            self.label_tensors[col] = torch.tensor(encoded, dtype=torch.long)
            self.num_classes[col] = len(le.classes_)
            self.fill_rates[col] = len(non_null) / n_items
            self.encoders[col] = le
            null_count = n_items - len(non_null)
            logger.info(f"  {col}: {len(le.classes_)} classes, {null_count} nulls ({null_count/n_items:.1%})")

        logger.info(f"Loaded {len(self.embeddings):,} items, {len(self.label_tensors)} attribute columns")

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        labels = {col: t[idx] for col, t in self.label_tensors.items()}
        return self.embeddings[idx], labels


def collate_with_labels(batch):
    embeddings = torch.stack([b[0] for b in batch])
    label_keys = batch[0][1].keys()
    labels = {k: torch.stack([b[1][k] for b in batch]) for k in label_keys}
    return embeddings, labels


# ── VQ components (copied from base, no changes needed) ───────────────────────

class QuantizationOutput(NamedTuple):
    quantized_st: Tensor
    quantized: Tensor
    indices: Tensor
    loss: Tensor
    codebook_loss: Optional[Tensor]
    commitment_loss: Tensor


class BaseVectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, codebook_embedding_dim: int,
                 commitment_weight: float, use_rotation_trick: bool):
        super().__init__()
        self.codebook_embedding_dim = codebook_embedding_dim
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.use_rotation_trick = use_rotation_trick

        self.embedding = nn.Embedding(codebook_size, codebook_embedding_dim)
        self.embedding.weight.data.uniform_(-1 / codebook_size, 1 / codebook_size)

        self.register_buffer("usage_count", torch.zeros(codebook_size))
        self.register_buffer("update_count", torch.tensor(0))

    @staticmethod
    def l2norm(t, dim=-1, eps=1e-6):
        return F.normalize(t, p=2, dim=dim, eps=eps)

    @staticmethod
    def safe_div(num, den, eps=1e-6):
        return num / den.clamp(min=eps)

    @staticmethod
    def rotation_trick(u, q, e):
        w = BaseVectorQuantizer.l2norm(u + q, dim=-1).detach()
        w_col, w_row = w.unsqueeze(-1), w.unsqueeze(-2)
        u_col = u.unsqueeze(-1).detach()
        q_row = q.unsqueeze(-2).detach()
        if e.ndim == 2:
            e_exp = e.unsqueeze(1)
            result = e_exp - 2 * (e_exp @ w_col @ w_row) + 2 * (e_exp @ u_col @ q_row)
            return result.squeeze(1)
        return e - 2 * (e @ w_col @ w_row).squeeze(-1) + 2 * (e @ u_col @ q_row).squeeze(-1)

    @staticmethod
    def rotate_to(src, tgt):
        orig_shape = src.shape
        src_flat = src.reshape(-1, src.shape[-1])
        tgt_flat = tgt.reshape(-1, tgt.shape[-1])
        norm_src = src_flat.norm(dim=-1, keepdim=True)
        norm_tgt = tgt_flat.norm(dim=-1, keepdim=True)
        rotated = BaseVectorQuantizer.rotation_trick(
            BaseVectorQuantizer.safe_div(src_flat, norm_src),
            BaseVectorQuantizer.safe_div(tgt_flat, norm_tgt),
            src_flat,
        )
        return (rotated * BaseVectorQuantizer.safe_div(norm_tgt, norm_src).detach()).reshape(orig_shape)

    def find_nearest_codes(self, x):
        flat_x = x.reshape(-1, self.codebook_embedding_dim)
        distances = torch.cdist(flat_x, self.embedding.weight)
        indices = distances.argmin(dim=1)
        quantized = self.embedding(indices).view(x.shape)
        return indices.view(x.shape[:-1]), quantized

    def apply_gradient_estimator(self, x, quantized):
        if self.training and x.requires_grad:
            if self.use_rotation_trick:
                return self.rotate_to(x, quantized)
            return x + (quantized - x).detach()
        return quantized

    def update_usage(self, indices):
        flat = indices.flatten()
        self.usage_count.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float))
        self.update_count += 1

    def get_usage_rate(self):
        if self.update_count == 0:
            return 0.0
        return (self.usage_count > 0).float().mean().item()

    def reset_usage_count(self):
        self.usage_count.zero_()


class VectorQuantizer(BaseVectorQuantizer):
    def forward(self, x):
        indices, quantized = self.find_nearest_codes(x)
        quantized_st = self.apply_gradient_estimator(x, quantized)
        codebook_loss = F.mse_loss(x.detach(), quantized)
        commitment_loss = F.mse_loss(x, quantized.detach())
        loss = codebook_loss + self.commitment_weight * commitment_loss
        if self.training:
            self.update_usage(indices)
        return QuantizationOutput(quantized_st, quantized, indices, loss, codebook_loss, commitment_loss)

    def reset_unused_codes(self, batch_data):
        if self.update_count == 0:
            return
        unused = (self.usage_count == 0).nonzero().squeeze(-1)
        if len(unused) > 0 and batch_data.shape[0] >= len(unused):
            flat = batch_data.reshape(-1, self.codebook_embedding_dim)
            rand_idx = torch.randperm(flat.shape[0], device=flat.device)[: len(unused)]
            self.embedding.weight.data[unused] = flat[rand_idx].detach()
        self.reset_usage_count()


# ── Supervised Contrastive Loss ───────────────────────────────────────────────

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., 2020).
    Positives = items in the batch sharing the same label.
    Negatives = all other items in the batch.

    Input:
        features: (B, D) — L2-normalised representations
        labels:   (B,)  — integer class labels; -1 = ignore (no supervision)
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: Tensor, labels: Tensor) -> Tensor:
        # Drop items with label == -1 (missing attribute)
        valid = labels != -1
        if valid.sum() < 2:
            return features.new_tensor(0.0)

        feats = F.normalize(features[valid], dim=-1)
        lbls  = labels[valid]
        B = feats.shape[0]

        # Similarity matrix
        sim = torch.matmul(feats, feats.T) / self.temperature  # (B, B)

        # Mask out self-comparisons
        self_mask = torch.eye(B, dtype=torch.bool, device=feats.device)
        sim = sim.masked_fill(self_mask, float("-inf"))

        # Positive mask: same label, not self
        pos_mask = (lbls.unsqueeze(0) == lbls.unsqueeze(1)) & ~self_mask  # (B, B)

        # Skip rows that have no positive pair in this batch
        has_pos = pos_mask.any(dim=1)
        if has_pos.sum() == 0:
            return features.new_tensor(0.0)

        # Log-sum-exp over all non-self pairs (denominator)
        log_denom = torch.logsumexp(sim, dim=1)  # (B,)

        # Mean log-prob over positive pairs.
        # Use masked_fill instead of float-mask multiply: 0.0 * -inf = NaN.
        log_prob_pos = sim - log_denom.unsqueeze(1)  # (B, B)
        log_prob_pos = log_prob_pos.masked_fill(~pos_mask, 0.0)
        mean_log_prob = log_prob_pos.sum(1) / pos_mask.float().sum(1).clamp(min=1)

        loss = -mean_log_prob[has_pos].mean()
        return loss


# ── Main model ────────────────────────────────────────────────────────────────

class RQVAEPurity(nn.Module):
    """
    RQ-VAE with a SupConLoss exclusivity term on the quantized representation
    — enforces between-cluster exclusivity for attribute combinations.
    """

    def __init__(self, config: RQVAEPurityConfig):
        super().__init__()
        self.config = config

        # Encoder
        dims = [config.item_embedding_dim] + config.encoder_hidden_dims + [config.codebook_embedding_dim]
        enc = []
        for i in range(len(dims) - 2):
            enc += [nn.Linear(dims[i], dims[i + 1]), nn.SiLU()]
        enc.append(nn.Linear(dims[-2], dims[-1]))
        self.encoder = nn.Sequential(*enc)

        # Decoder
        dims_r = [config.codebook_embedding_dim] + config.encoder_hidden_dims[::-1] + [config.item_embedding_dim]
        dec = []
        for i in range(len(dims_r) - 2):
            dec += [nn.Linear(dims_r[i], dims_r[i + 1]), nn.SiLU()]
        dec.append(nn.Linear(dims_r[-2], dims_r[-1]))
        self.decoder = nn.Sequential(*dec)

        # VQ layers
        self.vq_layers = nn.ModuleList([
            VectorQuantizer(
                config.codebook_size,
                config.codebook_embedding_dim,
                config.commitment_weight,
                config.use_rotation_trick,
            )
            for _ in range(config.codebook_quantization_levels)
        ])

        # Shared SupCon loss — parameter-free (just a temperature scalar), so one
        # instance is reused for every (level, combo) pair.
        self.supcon_loss = SupConLoss(temperature=config.supcon_temperature)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(
        self, x: Tensor, labels: Dict[str, Tensor]
    ) -> Tuple[Tensor, List[Tensor], dict]:
        z = self.encode(x)
        quantized_out = torch.zeros_like(z)
        residual = z

        all_indices, all_vq_outputs = [], []
        vq_loss = 0.0
        codebook_losses, commitment_losses = [], []

        for vq_layer in self.vq_layers:
            vq_out = vq_layer(residual)
            residual = residual - vq_out.quantized.detach()
            quantized_out = quantized_out + vq_out.quantized_st
            all_indices.append(vq_out.indices)
            all_vq_outputs.append(vq_out)
            vq_loss = vq_loss + vq_out.loss
            if vq_out.codebook_loss is not None:
                codebook_losses.append(vq_out.codebook_loss)
            commitment_losses.append(vq_out.commitment_loss)

        x_recon = self.decode(quantized_out)
        recon_loss = F.mse_loss(x_recon, x)

        # ── Exclusivity loss — hierarchical per-level SupCon ──────────────────
        # Level 0 → masterCategory (coarse separation of root buckets)
        # Level 1 → articleType    (product-type separation)
        # Level 2 → all articleType x <attr> combos (exclusivity for every
        #            attribute combination at once, not just colour)
        # Each SupCon operates on only that level's quantized vector, so
        # gradients stay local to that level's codebook — they don't bleed
        # into other levels. When a level has multiple combos, only the
        # combos with a valid positive pair *in this batch* are averaged, so
        # the effective per-combo pressure doesn't get diluted by combos that
        # happened to have no signal this step.
        excl_loss = x.new_tensor(0.0)
        excl_log: Dict[str, float] = {}
        for level, (vq_out, combo_list) in enumerate(zip(all_vq_outputs, self.config.supcon_labels_per_level)):
            q_level = vq_out.quantized_st   # straight-through: gradient flows back
            active_losses = []
            for combo_col in combo_list:
                if combo_col not in labels:
                    continue
                lv_loss = self.supcon_loss(q_level, labels[combo_col])
                excl_log[f"l{level}_{combo_col}"] = lv_loss.item()
                if (labels[combo_col] != -1).sum() >= 2:
                    active_losses.append(lv_loss)
            if active_losses:
                level_loss = torch.stack(active_losses).mean()
                excl_loss = excl_loss + level_loss
                excl_log[f"l{level}_mean"] = level_loss.item()

        total_loss = recon_loss + vq_loss + self.config.lambda_exclusivity * excl_loss

        loss_dict = {
            "loss":             total_loss,
            "recon_loss":       recon_loss,
            "vq_loss":          vq_loss,
            "excl_loss":        excl_loss,
            "excl_log":         excl_log,
            "codebook_losses":  codebook_losses,
            "commitment_losses": commitment_losses,
            "indices":          all_indices,
            "residual":         residual,
        }

        return x_recon, all_indices, loss_dict

    def encode_to_semantic_ids(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            z = self.encode(x)
            residual = z
            ids = []
            for vq_layer in self.vq_layers:
                vq_out = vq_layer(residual)
                ids.append(vq_out.indices)
                residual = residual - vq_out.quantized
            return torch.stack(ids, dim=-1)

    def calculate_unique_ids_proportion(self, semantic_ids: Tensor) -> float:
        B = semantic_ids.shape[0]
        if B <= 1:
            return 1.0
        a = semantic_ids.unsqueeze(1)
        b = semantic_ids.unsqueeze(0)
        matches = (a == b).all(dim=-1)
        has_dup = torch.triu(matches, diagonal=1).any(dim=1)
        return (~has_dup).sum().item() / B

    def calculate_codebook_usage(self) -> List[float]:
        return [vq.get_usage_rate() for vq in self.vq_layers]

    def calculate_avg_residual_norm(self, residual: Tensor) -> float:
        return residual.norm(dim=-1).mean().item()

    def kmeans_init(self, data_loader, device):
        logger.info("Initializing codebooks with k-means...")
        first_batch, _ = next(iter(data_loader))
        first_batch = first_batch.to(device)
        with torch.no_grad():
            z = self.encode(first_batch)
            residual = z
            for level, vq_layer in enumerate(self.vq_layers):
                residual_np = residual.cpu().numpy()
                km = KMeans(n_clusters=self.config.codebook_size, n_init=10, random_state=0)
                km.fit(residual_np)
                vq_layer.embedding.weight.data = torch.from_numpy(km.cluster_centers_).to(device)
                logger.info(f"  Level {level}: initialized {self.config.codebook_size} codes")
                if level < self.config.codebook_quantization_levels - 1:
                    vq_out = vq_layer(residual)
                    residual = residual - vq_out.quantized


# ── Training loop ─────────────────────────────────────────────────────────────

def get_gradient_norm(model: nn.Module) -> float:
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    return torch.norm(torch.stack([torch.norm(g, 2) for g in grads]), 2).item()


def get_loss(model: RQVAEPurity, val_loader: DataLoader, device: str) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for data, labels in val_loader:
            data = data.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}
            _, _, loss_dict = model(data, labels)
            total += loss_dict["loss"].item()
            count += 1
    return total / count


def evaluate(model, val_loader, train_loader, device, global_step, epoch):
    model.eval()
    with torch.no_grad():
        batch_data, batch_labels = next(iter(train_loader))
        batch_data = batch_data.to(device)
        batch_labels = {k: v.to(device) for k, v in batch_labels.items()}
        _, indices, loss_dict = model(batch_data, batch_labels)
        codebook_usage = model.calculate_codebook_usage()
        avg_residual_norm = model.calculate_avg_residual_norm(loss_dict["residual"])
        semantic_ids = torch.stack(indices, dim=-1)
        unique_ids = model.calculate_unique_ids_proportion(semantic_ids)

    val_loss = get_loss(model, val_loader, device)
    usage_str = "/".join(f"{u:.2f}" for u in codebook_usage)

    logger.info(
        f"Step {global_step:05d} | Epoch {epoch:05d} | val_loss: {val_loss:.2e} | "
        f"excl: {loss_dict['excl_loss'].item():.2e} | "
        f"codebook: {usage_str} | residual: {avg_residual_norm:.3f} | unique_ids: {unique_ids:.1%}"
    )

    log_dict = {
        "loss/validation":         val_loss,
        "metrics/avg_residual_norm": avg_residual_norm,
        "metrics/unique_ids":      unique_ids,
        "epoch": epoch, "step": global_step,
    }
    for level, u in enumerate(codebook_usage):
        log_dict[f"metrics/codebook_usage_level_{level}"] = u
    wandb.log(log_dict)

    return {
        "val_loss": val_loss,
        "codebook_usage": codebook_usage,
        "codebook_usage_str": usage_str,
        "avg_residual_norm": avg_residual_norm,
        "unique_ids_proportion": unique_ids,
    }


def save_checkpoint(model, optimizer, scheduler, metrics, config, global_step, epoch, best_loss):
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "epoch": epoch,
        "step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "val_loss": metrics["val_loss"],
        "config": config.__dict__,
    }
    torch.save(data, config.checkpoint_dir / f"checkpoint_step_{global_step}.pth")

    if metrics["val_loss"] < best_loss:
        best_loss = metrics["val_loss"]
        best_path = config.checkpoint_dir / "best_model.pth"
        torch.save(data, best_path)
        logger.info(f"New best val_loss: {best_loss:.4e} → {best_path}")
        artifact = wandb.Artifact(
            f"rqvae-purity-best-{config.category}", type="model",
            metadata={**metrics, "step": global_step, "epoch": epoch + 1},
        )
        artifact.add_file(str(best_path))
        wandb.log_artifact(artifact)

    return best_loss


def train_rqvae_purity(model: RQVAEPurity, data_loader, config: RQVAEPurityConfig,
                       device: str, val_loader=None):
    model = model.to(device)
    if config.use_kmeans_init:
        model.kmeans_init(data_loader, device)

    if device == "cuda":
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model)

    fused_ok = "fused" in inspect.signature(torch.optim.AdamW).parameters
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.max_lr, weight_decay=0.01,
        fused=fused_ok and device == "cuda",
    )

    steps_per_epoch = len(data_loader) // config.gradient_accumulation_steps
    total_steps = steps_per_epoch * config.num_epochs

    if config.scheduler_type == "cosine_with_warmup":
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=config.warmup_start_lr / config.max_lr,
            total_iters=config.warmup_steps,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - config.warmup_steps, eta_min=config.min_lr
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[config.warmup_steps]
        )
    elif config.scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=config.min_lr
        )
    else:
        scheduler = None

    best_loss = float("inf")
    global_step = 0

    for epoch in range(config.num_epochs):
        model.train()

        for batch_idx, (data, labels) in enumerate(data_loader):
            if batch_idx % config.gradient_accumulation_steps == 0:
                t0 = time.time()
                optimizer.zero_grad()
                loss_accum = 0.0
                excl_accum   = 0.0

            data = data.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}

            _, indices, loss_dict = model(data, labels)
            loss = loss_dict["loss"] / config.gradient_accumulation_steps
            loss.backward()

            loss_accum   += loss_dict["loss"].detach().item()
            excl_accum   += loss_dict["excl_loss"].detach().item()

            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                grad_norm_before = get_gradient_norm(model)
                if config.use_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                grad_norm_after = get_gradient_norm(model)

                optimizer.step()
                if scheduler:
                    scheduler.step()

                t1 = time.time()
                global_step += 1
                n = config.gradient_accumulation_steps
                avg_loss   = loss_accum / n
                avg_excl   = excl_accum / n

                if global_step == 1 or global_step % config.steps_per_train_log == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    usage = model.calculate_codebook_usage()
                    semantic_ids = torch.stack(loss_dict["indices"], dim=-1)
                    unique_ids = model.calculate_unique_ids_proportion(semantic_ids)
                    usage_str = "/".join(f"{u:.2f}" for u in usage)

                    logger.info(
                        f"Step {global_step:05d} | Epoch {epoch+1:05d} | lr: {lr:.2e} | "
                        f"loss: {avg_loss:.2e} | recon: {loss_dict['recon_loss'].item():.2e} | "
                        f"vq: {loss_dict['vq_loss'].item():.2e} | "
                        f"excl: {avg_excl:.2e} | codebook: {usage_str} | unique: {unique_ids:.1%} | "
                        f"time: {(t1-t0)*1000:.0f}ms"
                    )

                    log_dict = {
                        "loss/total":          avg_loss,
                        "loss/reconstruction": loss_dict["recon_loss"].item(),
                        "loss/vq":             loss_dict["vq_loss"].item(),
                        "loss/exclusivity":    avg_excl,
                        "metrics/lr":          lr,
                        "metrics/grad_norm":   grad_norm_before,
                        "metrics/unique_ids":  unique_ids,
                        "epoch": epoch + 1, "step": global_step,
                    }
                    for level, u in enumerate(usage):
                        log_dict[f"metrics/codebook_usage_level_{level}"] = u
                    for key, v in loss_dict.get("excl_log", {}).items():
                        log_dict[f"loss/excl_{key}"] = v
                    wandb.log(log_dict)

                if global_step % config.steps_per_val_log == 0 and val_loader:
                    metrics = evaluate(model, val_loader, data_loader, device, global_step, epoch + 1)
                    best_loss = save_checkpoint(
                        model, optimizer, scheduler, metrics, config, global_step, epoch, best_loss
                    )
                    model.train()

                if config.reset_unused_codes and global_step % config.steps_per_codebook_reset == 0:
                    if not (config.scheduler_type == "cosine_with_warmup" and global_step < config.warmup_steps):
                        model.eval()
                        usage = model.calculate_codebook_usage()
                        reset_batch, _ = next(iter(data_loader))
                        for level, vq_layer in enumerate(model.vq_layers):
                            if isinstance(vq_layer, VectorQuantizer):
                                if usage[level] < config.codebook_usage_threshold:
                                    with torch.no_grad():
                                        z = model.encode(reset_batch.to(device))
                                        res = z
                                        for i in range(level):
                                            vq_out = model.vq_layers[i](res)
                                            res = res - vq_out.quantized
                                        vq_layer.reset_unused_codes(res)
                                vq_layer.reset_usage_count()
                        model.train()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config = RQVAEPurityConfig()
    device_manager = DeviceManager(logger)
    device = device_manager.device

    dataset = EmbeddingDatasetWithLabels(
        str(config.embeddings_path),
        attr_cols=ATTR_COLS,
    )

    val_size = int(len(dataset) * config.val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=16, pin_memory=device_manager.supports_pin_memory,
        prefetch_factor=8, persistent_workers=True, drop_last=False,
        collate_fn=collate_with_labels,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=4, pin_memory=device_manager.supports_pin_memory,
        prefetch_factor=4, persistent_workers=True,
        collate_fn=collate_with_labels,
    )

    model = RQVAEPurity(config)

    run_name = (
        f"rqvae-purity-L{config.codebook_quantization_levels}"
        f"-C{config.codebook_size}-D{config.codebook_embedding_dim}"
        f"-le{config.lambda_exclusivity}"
    )
    import os as _os
    _wandb_mode = "offline" if not _os.environ.get("WANDB_API_KEY") else "online"
    wandb.init(project="rqvae", name=run_name, config=config.__dict__, mode=_wandb_mode)

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_rqvae_purity(model, train_loader, config, device, val_loader)

    final_path = config.checkpoint_dir / "final_model.pth"
    torch.save({"model_state_dict": model.state_dict(), "config": config.__dict__}, final_path)
    logger.info(f"Training complete → {final_path}")
