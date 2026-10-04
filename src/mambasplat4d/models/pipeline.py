"""ClassificationPipeline: spatial + temporal + head.

Modes from cfg.training.{heads,freeze}:
  per_frame_mode   heads.frame_aux AND freeze.classifier: frame logits primary.
  has_aux_losses   heads.frame_aux AND NOT freeze.classifier: sequence logits + aux frame outputs.
  neither          sequence logits only.

get_param_groups: differential LR if any optimizer.{spatial_lr,bridge_lr,temporal_lr,head_lr}
set, else single group at optimizer.lr.
"""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from omegaconf import DictConfig

from .registry import build_spatial_encoder, build_temporal_encoder
from .heads.classifier import ClassificationHead, ContrastiveHead
from .temporal.vn_bridge import VNInBridge
from ..utils.feature_config import get_invariant_dim


def _resolve_freeze_flags(cfg: DictConfig) -> dict:
    """Freeze flags {spatial, bridge, temporal, classifier, frame_heads}.

    Precedence: training.freeze.<name> > training.{spatial_freeze,bridge_freeze}
    > spatial.freeze > False. freeze.classifier forced False when
    loss.temporal_clip_classification > 0.
    """
    train = cfg.get("training", {})
    freeze_cfg = train.get("freeze", {}) or {}

    def _explicit(name):
        v = freeze_cfg.get(name, None)
        return None if v is None else bool(v)

    spatial = _explicit("spatial")
    if spatial is None:
        legacy = train.get("spatial_freeze", None)
        spatial = (
            bool(legacy)
            if legacy is not None
            else bool(cfg.spatial.get("freeze", False))
        )

    bridge = _explicit("bridge")
    if bridge is None:
        bridge = bool(train.get("bridge_freeze", False))

    temporal = bool(freeze_cfg.get("temporal", False))
    classifier = bool(freeze_cfg.get("classifier", False))
    frame_heads = bool(freeze_cfg.get("frame_heads", False))

    # clip loss needs trainable sequence classifier
    clip_w = float(train.get("loss", {}).get("temporal_clip_classification", 0))
    if classifier and clip_w > 0:
        classifier = False

    return {
        "spatial": spatial,
        "bridge": bridge,
        "temporal": temporal,
        "classifier": classifier,
        "frame_heads": frame_heads,
    }


def _resolve_output_flags(cfg: DictConfig) -> dict:
    """Output flags: per_frame_mode (frame head primary, seq classifier frozen),
    has_aux_losses (frame heads auxiliary). clip_logits/h_seq always emitted when T > 1."""
    train = cfg.get("training", {})
    heads_cfg = train.get("heads", {}) or {}
    freeze_cfg = train.get("freeze", {}) or {}

    frame_aux = bool(heads_cfg.get("frame_aux", False))
    classifier_frozen = bool(freeze_cfg.get("classifier", False))
    return {
        "per_frame_mode": frame_aux and classifier_frozen,
        "has_aux_losses": frame_aux and not classifier_frozen,
    }


class ClassificationPipeline(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        out_flags = _resolve_output_flags(cfg)
        self.per_frame_mode = out_flags["per_frame_mode"]
        self.has_aux_losses = out_flags["has_aux_losses"]

        # aux heads gated so CE-only recipes carry no dead params
        heads_cfg = cfg.get("training", {}).get("heads", {}) or {}
        self.use_frame_aux = bool(heads_cfg.get("frame_aux", False))
        self.use_norm_aux = bool(heads_cfg.get("norm_aux", False))

        self.spatial_encoder = build_spatial_encoder(cfg.spatial)

        # standalone bridge (bridge=vn_bridge)
        bridge_cfg = cfg.get("bridge", None)
        self.bridge_mode = bridge_cfg is not None
        if self.bridge_mode:
            dim_invariant = get_invariant_dim(bridge_cfg.feature_mode)
            self.bridge = VNInBridge(
                C=bridge_cfg.C,
                dim_invariant=dim_invariant,
                d_mamba=bridge_cfg.hidden_dim,
                frame_hidden=bridge_cfg.get("frame_hidden", None),
            )
            self.spatial_encoder.return_vectors = True
            self.spatial_encoder.classifier_head.projection = nn.Identity()

            if bridge_cfg.get("pretrained_ckpt", None):
                self._load_partial(self.bridge, bridge_cfg.pretrained_ckpt)
            if bridge_cfg.get("freeze", False):
                self.bridge.requires_grad_(False)

        self.temporal_encoder = build_temporal_encoder(cfg.temporal)

        embed_dim = cfg.embed_dim
        num_classes = cfg.num_classes

        # vn_mamba: VN-In bridge + Mamba on post-pool (V, X_inv) -> (B, d_mamba); name check avoids mamba_ssm import
        self.vn_mamba_mode = cfg.temporal.name == "vn_mamba"

        if self.vn_mamba_mode:
            head_dim = getattr(
                self.temporal_encoder, "output_dim", cfg.temporal.hidden_dim
            )
            self.spatial_encoder.return_vectors = True
            # drop unused projection (issue 3)
            self.spatial_encoder.classifier_head.projection = nn.Identity()
            if self.use_norm_aux:
                self.norm_classifier = ClassificationHead(
                    embed_dim=cfg.spatial.dim, num_classes=num_classes
                )
        else:
            head_dim = cfg.temporal.hidden_dim

        self.classifier = ClassificationHead(
            embed_dim=head_dim, num_classes=num_classes
        )

        if self.use_frame_aux:
            if self.vn_mamba_mode:
                # frame heads see bridge output (d_mamba)
                frame_head_dim = cfg.temporal.hidden_dim
            else:
                frame_head_dim = embed_dim
            self.frame_classifier = ClassificationHead(
                embed_dim=frame_head_dim, num_classes=num_classes
            )
            self.frame_contrastive = ContrastiveHead(embed_dim=frame_head_dim)

        freeze = _resolve_freeze_flags(cfg)

        # frozen random bridge silently poisons eval
        temporal_ckpt = cfg.temporal.get("pretrained_ckpt", None)
        if self.vn_mamba_mode and freeze["bridge"] and not temporal_ckpt:
            raise ValueError(
                "training.freeze.bridge=true requires temporal.pretrained_ckpt "
                "to avoid freezing a randomly initialized bridge."
            )

        # load before freezing
        if cfg.spatial.get("pretrained_ckpt", None):
            self._load_partial(self.spatial_encoder, cfg.spatial.pretrained_ckpt)
        if cfg.temporal.get("pretrained_ckpt", None):
            self._load_partial(self.temporal_encoder, cfg.temporal.pretrained_ckpt)

        if freeze["spatial"]:
            self.spatial_encoder.requires_grad_(False)
        if cfg.temporal.get("freeze", False):
            # independent of training.freeze.temporal
            self.temporal_encoder.requires_grad_(False)
        if freeze["bridge"]:
            if self.vn_mamba_mode:
                self.temporal_encoder.bridge.requires_grad_(False)
            elif self.bridge_mode:
                self.bridge.requires_grad_(False)
        if freeze["temporal"]:
            if self.vn_mamba_mode:
                # Mamba + pos-enc only; bridge via freeze.bridge
                for p in self.temporal_encoder.mamba.parameters():
                    p.requires_grad = False
                for p in self.temporal_encoder.pos_encoder.parameters():
                    p.requires_grad = False
            else:
                self.temporal_encoder.requires_grad_(False)
        if freeze["classifier"]:
            self.classifier.requires_grad_(False)
        if freeze["frame_heads"]:
            if hasattr(self, "frame_classifier"):
                self.frame_classifier.requires_grad_(False)
            if hasattr(self, "frame_contrastive"):
                self.frame_contrastive.requires_grad_(False)
            if hasattr(self, "norm_classifier"):
                self.norm_classifier.requires_grad_(False)

    @staticmethod
    def _load_partial(module: nn.Module, ckpt_path: str):
        """Load checkpoint; strip pipeline prefix with best key overlap."""
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)

        module_keys = set(module.state_dict().keys())
        best_state = state
        best_overlap = len(set(state.keys()) & module_keys)

        prefix_candidates = [
            f"{name}." for name in ("spatial_encoder", "temporal_encoder", "bridge")
        ]
        for prefix in prefix_candidates:
            matching = {
                k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)
            }
            overlap = len(set(matching.keys()) & module_keys)
            if overlap > best_overlap:
                best_state = matching
                best_overlap = overlap

        module.load_state_dict(best_state, strict=False)

    def forward(
        self,
        gaussian_data: dict,
        mask: Optional[Tensor] = None,
        T: int = 1,
    ) -> dict:
        """gaussian_data (B*T, N, ...), mask (B*T, N), T timesteps (1 = static).

        Returns by mode:
          per_frame_mode: {embeddings, logits(frame), projections, norm_logits, [clip_logits, h_seq if T>1]}
          has_aux_losses: {embeddings, logits(seq), frame_logits, frame_projections}
          else:           {embeddings, logits(seq)}
        """
        BT = list(gaussian_data.values())[0].shape[0]
        B = BT // T

        frame_out = self.spatial_encoder(gaussian_data, mask=mask)

        if self.vn_mamba_mode:
            frame_vec, X_inv = frame_out  # (B*T, C, 3), (B*T, C, F)

            if self.per_frame_mode:
                # bridge only, no Mamba
                frame_emb = self.temporal_encoder.bridge(
                    frame_vec, X_inv, gaussian_data, mask
                )  # (B*T, d_mamba)
                norms = frame_vec.norm(dim=-1)  # (B*T, C)
                result = {
                    "embeddings": frame_emb,
                    "logits": self.frame_classifier(frame_emb),
                    "projections": self.frame_contrastive(frame_emb),
                }
                if self.use_norm_aux:
                    result["norm_logits"] = self.norm_classifier(norms)
                # clip logits + smoothness signal; losses gate on weights
                if T > 1:
                    h_seq = frame_emb.reshape(B, T, -1)  # (B, T, d_mamba)
                    clip_emb = h_seq.mean(dim=1)  # (B, d_mamba)
                    result["clip_logits"] = self.classifier(clip_emb)  # (B, K)
                    result["h_seq"] = h_seq  # smoothness loss
                return result

            features, h_frames = self.temporal_encoder(
                frame_vec,
                mask=mask,
                X_inv=X_inv,
                gaussian_data=gaussian_data,
                T=T,
            )  # features: (B, d_mamba), h_frames: (B*T, d_mamba)

            result = {"embeddings": features, "logits": self.classifier(features)}
            if self.has_aux_losses:
                result["frame_logits"] = self.frame_classifier(h_frames)
                result["frame_projections"] = self.frame_contrastive(h_frames)
            return result

        # scalar path
        frame_emb = frame_out  # (B*T, embed_dim)

        if self.per_frame_mode:
            result = {
                "embeddings": frame_emb,
                "logits": self.frame_classifier(frame_emb),
                "projections": self.frame_contrastive(frame_emb),
            }
            if T > 1:
                h_seq = frame_emb.reshape(B, T, -1)
                clip_emb = h_seq.mean(dim=1)
                result["clip_logits"] = self.classifier(clip_emb)
                result["h_seq"] = h_seq
            return result

        seq_emb = frame_emb.reshape(B, T, -1)  # (B, T, d)
        features = self.temporal_encoder(seq_emb)  # (B, hidden_dim)

        result = {"embeddings": features, "logits": self.classifier(features)}
        if self.has_aux_losses:
            result["frame_logits"] = self.frame_classifier(frame_emb)
            result["frame_projections"] = self.frame_contrastive(frame_emb)
        return result

    def get_param_groups(self, cfg: DictConfig) -> list:
        """Optimizer param groups.

        Differential LR (one group per component) if any optimizer.{spatial_lr,
        bridge_lr, temporal_lr, head_lr} set, else single LR.
        optimizer.no_decay_params (default True) splits each group via
        split_no_decay: biases, 1-D norm weights, ``_no_weight_decay`` params
        (mamba ``A_log``, ``D``) skip weight decay.
        """
        from ..utils.optim_groups import split_no_decay

        opt = cfg.training.optimizer
        no_decay_split = bool(opt.get("no_decay_params", True))
        wd = float(opt.get("weight_decay", 0.0))

        def _component(params, lr):
            """Group(s) for one component."""
            if no_decay_split:
                return split_no_decay(params, lr, wd)
            params_list = [p for p in params if p.requires_grad]
            return [{"params": params_list, "lr": lr}]

        diff_lr_keys = ("spatial_lr", "bridge_lr", "temporal_lr", "head_lr")
        if any(k in opt for k in diff_lr_keys):
            # aux heads optional
            head_params = list(self.classifier.parameters())
            if hasattr(self, "frame_classifier"):
                head_params += list(self.frame_classifier.parameters())
            if hasattr(self, "frame_contrastive"):
                head_params += list(self.frame_contrastive.parameters())
            if hasattr(self, "norm_classifier"):
                head_params += list(self.norm_classifier.parameters())

            groups = []
            groups += _component(self.spatial_encoder.parameters(), opt.spatial_lr)
            if self.vn_mamba_mode:
                groups += _component(
                    self.temporal_encoder.bridge_parameters(),
                    opt.bridge_lr,
                )
                groups += _component(
                    self.temporal_encoder.mamba_parameters(),
                    opt.temporal_lr,
                )
            else:
                groups += _component(
                    self.temporal_encoder.parameters(),
                    opt.temporal_lr,
                )
            groups += _component(head_params, opt.head_lr)
            return groups

        return _component(self.trainable_parameters(), opt.lr)

    def trainable_parameters(self):
        """Params with requires_grad."""
        return (p for p in self.parameters() if p.requires_grad)
