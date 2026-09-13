#!/usr/bin/env python3
"""
Verification Script: Layer Freezing Claims

This script verifies the paper claims about layer freezing:
- Paper claims: 24 transformer blocks, freeze blocks 0-11, train blocks 12-23
- Paper claims: 82% frozen (252M/307M), 18% trainable

Run: python train/verify_layer_freezing.py
"""

import torch
from sentence_transformers import SentenceTransformer

def main():
    print("=" * 70)
    print("LAYER FREEZING VERIFICATION SCRIPT")
    print("=" * 70)

    # 1. Load model exactly as training code does
    print("\n[1] Loading model (same as train_trigger.py:125-128)...")
    model = SentenceTransformer(
        "google/embeddinggemma-300m",
        model_kwargs={"torch_dtype": torch.float32}
    )

    # Access underlying transformer (same as train_trigger.py:145)
    base_model = model[0].auto_model

    # 2. Print architecture details
    print("\n" + "=" * 70)
    print("MODEL ARCHITECTURE")
    print("=" * 70)
    print(f"Model type: {type(base_model).__name__}")
    print(f"Hidden size: {base_model.config.hidden_size}")
    print(f"Num hidden layers: {base_model.config.num_hidden_layers}")
    print(f"Num attention heads: {base_model.config.num_attention_heads}")
    print(f"Vocab size: {base_model.config.vocab_size}")
    if hasattr(base_model.config, 'intermediate_size'):
        print(f"Intermediate size: {base_model.config.intermediate_size}")

    num_layers = len(base_model.layers)
    print(f"\nActual number of transformer layers: {num_layers}")
    print(f"Layer indices: 0 to {num_layers - 1}")

    # 3. Count parameters BEFORE freezing
    print("\n" + "=" * 70)
    print("PARAMETERS BEFORE FREEZING")
    print("=" * 70)

    total_before = sum(p.numel() for p in model.parameters())
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_before:,}")
    print(f"Trainable params: {trainable_before:,}")

    # 4. Detailed parameter breakdown
    print("\n" + "=" * 70)
    print("PARAMETER BREAKDOWN BY COMPONENT")
    print("=" * 70)

    # Embedding layer
    embed_params = sum(p.numel() for p in base_model.embed_tokens.parameters())
    print(f"\nembed_tokens: {embed_params:,} params ({100*embed_params/total_before:.1f}%)")

    # Each transformer layer
    print(f"\nTransformer layers ({num_layers} total):")
    layer_params_list = []
    for i, layer in enumerate(base_model.layers):
        lp = sum(p.numel() for p in layer.parameters())
        layer_params_list.append(lp)
        if i < 3 or i >= num_layers - 3:
            print(f"  Layer {i:2d}: {lp:,} params")
        elif i == 3:
            print("  ...")

    total_layer_params = sum(layer_params_list)
    params_per_layer = layer_params_list[0] if layer_params_list else 0
    print(f"\nTotal transformer layer params: {total_layer_params:,}")
    print(f"Params per layer: {params_per_layer:,} (assuming uniform)")

    # Final norm if present
    other_params = total_before - embed_params - total_layer_params
    if other_params > 0:
        print(f"\nOther params (norm, pooling, etc.): {other_params:,}")

    # 5. Apply freezing (same as train_trigger.py:148-155)
    print("\n" + "=" * 70)
    print("APPLYING FREEZING (same as training code)")
    print("=" * 70)

    # Freeze embedding layer
    print("\nFreezing embed_tokens...")
    for param in base_model.embed_tokens.parameters():
        param.requires_grad = False

    # Freeze layers 0-11 (same as train_trigger.py:152-155)
    freeze_layers = 12
    print(f"Freezing layers 0-{freeze_layers-1}...")
    for i in range(freeze_layers):
        for param in base_model.layers[i].parameters():
            param.requires_grad = False

    # 6. Count parameters AFTER freezing
    print("\n" + "=" * 70)
    print("PARAMETERS AFTER FREEZING")
    print("=" * 70)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"Total params: {total_params:,}")
    print(f"Frozen params: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")
    print(f"Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")

    # 7. Verify what's frozen vs trainable
    print("\n" + "=" * 70)
    print("DETAILED FROZEN/TRAINABLE STATUS")
    print("=" * 70)

    # Check embed_tokens
    embed_trainable = sum(p.numel() for p in base_model.embed_tokens.parameters() if p.requires_grad)
    embed_frozen = embed_params - embed_trainable
    print(f"\nembed_tokens: {embed_frozen:,} frozen, {embed_trainable:,} trainable")

    # Check each layer
    print(f"\nLayer-by-layer status:")
    frozen_layer_params = 0
    trainable_layer_params = 0
    for i, layer in enumerate(base_model.layers):
        layer_trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
        layer_frozen = layer_params_list[i] - layer_trainable
        status = "FROZEN" if layer_trainable == 0 else "TRAINABLE"
        if i < freeze_layers:
            frozen_layer_params += layer_params_list[i]
        else:
            trainable_layer_params += layer_params_list[i]
        if i < 3 or i >= num_layers - 3 or i == freeze_layers - 1 or i == freeze_layers:
            print(f"  Layer {i:2d}: {status}")
        elif i == 3:
            print("  ...")

    # 8. Summary comparison with paper claims
    print("\n" + "=" * 70)
    print("VERIFICATION: PAPER CLAIMS vs ACTUAL")
    print("=" * 70)

    paper_total = 307_000_000
    paper_frozen = 252_000_000
    paper_trainable = 55_000_000
    paper_frozen_pct = 82
    paper_trainable_pct = 18
    paper_num_blocks = 24
    paper_frozen_blocks = "0-11"
    paper_trainable_blocks = "12-23"

    actual_frozen_pct = 100 * frozen_params / total_params
    actual_trainable_pct = 100 * trainable_params / total_params
    actual_trainable_blocks = f"12-{num_layers - 1}"

    print(f"\n{'Metric':<25} {'Paper Claims':<20} {'Actual':<20} {'Match?'}")
    print("-" * 70)
    print(f"{'Total params':<25} {paper_total:,}{'':>3} {total_params:,}{'':>3} {'~' if abs(total_params - paper_total) < 10_000_000 else 'X'}")
    print(f"{'Frozen params':<25} {paper_frozen:,}{'':>3} {frozen_params:,}{'':>3} {'~' if abs(frozen_params - paper_frozen) < 10_000_000 else 'X'}")
    print(f"{'Trainable params':<25} {paper_trainable:,}{'':>3} {trainable_params:,}{'':>3} {'~' if abs(trainable_params - paper_trainable) < 10_000_000 else 'X'}")
    print(f"{'Frozen %':<25} {paper_frozen_pct}%{'':>16} {actual_frozen_pct:.1f}%{'':>14} {'OK' if abs(actual_frozen_pct - paper_frozen_pct) < 2 else 'X'}")
    print(f"{'Trainable %':<25} {paper_trainable_pct}%{'':>16} {actual_trainable_pct:.1f}%{'':>14} {'OK' if abs(actual_trainable_pct - paper_trainable_pct) < 2 else 'X'}")
    print(f"{'Num transformer blocks':<25} {paper_num_blocks}{'':>18} {num_layers}{'':>18} {'OK' if num_layers == paper_num_blocks else 'X'}")
    print(f"{'Frozen blocks':<25} {paper_frozen_blocks}{'':>17} 0-11{'':>15} OK")
    print(f"{'Trainable blocks':<25} {paper_trainable_blocks}{'':>14} {actual_trainable_blocks}{'':>14} {'OK' if num_layers == 24 else 'X'}")
    print(f"{'Embeddings frozen':<25} Yes{'':>17} Yes{'':>16} OK")

    # 9. Final verdict
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    issues = []
    if num_layers != 24:
        issues.append(f"- Model has {num_layers} blocks, not 24 as paper claims")
        issues.append(f"  -> Trainable blocks are 12-{num_layers-1}, not 12-23")
    if abs(total_params - paper_total) >= 10_000_000:
        issues.append(f"- Total params differ: {total_params:,} vs paper's {paper_total:,}")
    if abs(frozen_params - paper_frozen) >= 10_000_000:
        issues.append(f"- Frozen params differ: {frozen_params:,} vs paper's {paper_frozen:,}")
    if abs(actual_frozen_pct - paper_frozen_pct) >= 2:
        issues.append(f"- Frozen % differs: {actual_frozen_pct:.1f}% vs paper's {paper_frozen_pct}%")

    if not issues:
        print("\nALL CLAIMS VERIFIED CORRECTLY!")
    else:
        print("\nISSUES FOUND:")
        for issue in issues:
            print(issue)

        print("\n" + "-" * 70)
        print("SUGGESTED PAPER TEXT FIX:")
        print("-" * 70)
        print(f"""
Current paper text:
  "freeze embeddings + blocks 0-11; fine-tune blocks 12-23;
   82% frozen (252M/307M)"

Suggested correction:
  "freeze embeddings + blocks 0-11; fine-tune blocks 12-{num_layers-1};
   {actual_frozen_pct:.0f}% frozen ({frozen_params/1e6:.0f}M/{total_params/1e6:.0f}M)"
""")

    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()