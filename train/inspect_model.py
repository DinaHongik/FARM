#!/usr/bin/env python3
"""Inspect EmbeddingGemma model architecture and layers."""

from transformers import AutoModel, AutoTokenizer
import torch

def inspect_model():
    model_name = "google/embeddinggemma-300m"
    print(f"Loading model: {model_name}")

    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    # Basic info
    print(f"\n{'='*60}")
    print("MODEL ARCHITECTURE")
    print(f"{'='*60}")
    print(f"Model type: {type(model).__name__}")
    print(f"Hidden size: {model.config.hidden_size}")
    print(f"Num layers: {model.config.num_hidden_layers}")
    print(f"Num attention heads: {model.config.num_attention_heads}")
    print(f"Vocab size: {model.config.vocab_size}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Layer breakdown
    print(f"\n{'='*60}")
    print("LAYER BREAKDOWN")
    print(f"{'='*60}")

    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"\n{name}: {type(module).__name__} ({params:,} params)")

        # Show sub-modules
        for sub_name, sub_module in module.named_children():
            sub_params = sum(p.numel() for p in sub_module.parameters())
            print(f"  |- {sub_name}: {type(sub_module).__name__} ({sub_params:,} params)")

            # For transformer layers, show first few
            if hasattr(sub_module, '__len__') and len(list(sub_module.children())) > 0:
                children = list(sub_module.named_children())
                for i, (layer_name, layer) in enumerate(children[:3]):
                    layer_params = sum(p.numel() for p in layer.parameters())
                    print(f"      |- {layer_name}: {type(layer).__name__} ({layer_params:,} params)")
                if len(children) > 3:
                    print(f"      |- ... ({len(children) - 3} more layers)")

    # Detailed layer structure
    print(f"\n{'='*60}")
    print("DETAILED TRANSFORMER LAYERS")
    print(f"{'='*60}")

    # Find transformer layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
        print(f"\nNumber of transformer layers: {len(layers)}")

        # Inspect first layer structure
        first_layer = layers[0]
        print(f"\nFirst layer structure ({type(first_layer).__name__}):")
        for name, sub in first_layer.named_children():
            params = sum(p.numel() for p in sub.parameters())
            print(f"  |- {name}: {type(sub).__name__} ({params:,} params)")

            # Show deeper structure
            for n2, s2 in sub.named_children():
                p2 = sum(p.numel() for p in s2.parameters())
                print(f"      |- {n2}: {type(s2).__name__} ({p2:,} params)")

    # Freezing suggestions
    print(f"\n{'='*60}")
    print("FREEZING SUGGESTIONS")
    print(f"{'='*60}")
    print("""
Option 1: Freeze embeddings only
  - Freeze: embed_tokens
  - Train: all transformer layers

Option 2: Freeze lower layers (recommended)
  - Freeze: embed_tokens + layers 0-15 (first 16 layers)
  - Train: layers 16-25 (last 10 layers)

Option 3: Freeze most, train top only
  - Freeze: embed_tokens + layers 0-23
  - Train: layers 24-25 (last 2 layers only)

Option 4: LoRA (add adapters, don't modify weights)
  - Freeze: everything
  - Train: small adapter matrices (rank 8-16)
""")

if __name__ == "__main__":
    inspect_model()
