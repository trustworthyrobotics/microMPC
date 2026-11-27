"""Construct the visual Koopman architecture saved in rope checkpoints."""

import argparse
from torch import nn
from .models import KoopmanDynamics, VisualKoopmanModel, ProjectionHead


def build_model(args: argparse.Namespace) -> VisualKoopmanModel:
    """Build the saved architecture without downloading pretrained weights."""
    from transformers import ViTConfig, ViTModel

    config = ViTConfig(
        patch_size=args.patch_size,
        image_size=args.img_size,
        hidden_size=args.vit_hidden_size,
        num_hidden_layers=args.vit_depth,
        num_attention_heads=args.vit_num_heads,
        intermediate_size=args.vit_mlp_dim,
    )
    encoder = ViTModel(config, use_mask_token=False, add_pooling_layer=False)
    encoder.config.interpolate_pos_encoding = True
    projector = ProjectionHead(
        input_dim=args.vit_hidden_size,
        output_dim=args.embed_dim,
        hidden_dim=2048,
        norm_fn=nn.BatchNorm1d,
    )
    koopman = KoopmanDynamics(
        state_dim=args.markov_state_dim,
        control_dim=args.effective_action_dim,
        latent_dim=args.lifting_dim,
        hidden_width=args.lifting_hidden_width,
        depth=args.lifting_depth,
        dropout=args.lifting_dropout,
        activation_fn=nn.GELU,
    )
    nn.init.eye_(koopman.A.weight)
    nn.init.xavier_uniform_(koopman.B.weight)
    return VisualKoopmanModel(
        encoder=encoder,
        predictor=koopman,
        action_encoder=nn.Identity(),
        projector=projector,
        pred_proj=nn.Identity(),
    )
