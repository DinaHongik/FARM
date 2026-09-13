"""
LoRA Ablation Study for FARM Contrastive Training

Compares different adaptation strategies:
- Full fine-tuning (baseline)
- LoRA with ranks 8, 16, 32
- Layer freezing (our approach)

Comprehensive Evaluation Metrics (from evaluate_rag.py):
- Service-level: R@1, R@2, R@3, R@4, R@5, MRR@3, MRR@5
- Schema-level: Schema_R@1, Schema_R@3, Schema_R@5, Schema_MRR@3, Schema_MRR@5

Saves all results to JSON for paper analysis.
"""

import json
import time
import torch
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict, field
from copy import deepcopy

from sentence_transformers import SentenceTransformer, losses
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from datasets import Dataset

# Try to import PEFT for LoRA
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("WARNING: PEFT not installed. Install with: pip install peft")

# Local imports - handle both module and direct execution
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from .config import config
    from .dataset import build_trigger_pairs, build_action_pairs
except ImportError:
    from train.config import config
    from train.dataset import build_trigger_pairs, build_action_pairs


# =============================================================================
# METRIC COMPUTATION (from evaluate_rag.py - no hardcoding)
# =============================================================================

def compute_recall_at_k(ranks: List[int], k: int) -> float:
    """
    Compute Recall@K: fraction where correct item is in top-K.

    Formula: R@K = (# queries where correct item rank <= K) / (total # queries)

    Reference: TREC, MS MARCO, BEIR evaluation protocols
    """
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r <= k) / len(ranks)


def compute_mrr_at_k(ranks: List[int], k: int) -> float:
    """
    Compute MRR@K: Mean Reciprocal Rank for top-K.

    Formula: MRR@K = (1/Q) * Σ(1/rank_i) for rank_i <= K

    Reference: https://en.wikipedia.org/wiki/Mean_reciprocal_rank
    """
    if not ranks:
        return 0.0
    rr_sum = 0.0
    for r in ranks:
        if r <= k:
            rr_sum += 1.0 / r
    return rr_sum / len(ranks)


def normalize_name(name: str) -> str:
    """Normalize service name for comparison."""
    return name.lower().strip()


def get_schema_key(api_info: Dict, is_trigger: bool = True) -> str:
    """
    Extract a unique schema key from api_info.

    For triggers: uses Ingredients keys
    For actions: uses Action fields keys

    This ensures we match the EXACT API variant, not just service name.
    """
    if is_trigger:
        ingredients = api_info.get("Ingredients", {})
        keys = sorted(ingredients.keys()) if isinstance(ingredients, dict) else []
    else:
        fields = api_info.get("Action fields", {})
        keys = sorted(fields.keys()) if isinstance(fields, dict) else []
    return "|".join(keys).lower()


# =============================================================================
# COMPREHENSIVE METRICS DATACLASS
# =============================================================================

@dataclass
class RetrievalMetrics:
    """Comprehensive retrieval metrics for a single encoder (trigger or action)."""
    # Service-level metrics (TARGE-compatible)
    recall_at_1: float = 0.0
    recall_at_2: float = 0.0
    recall_at_3: float = 0.0
    recall_at_4: float = 0.0
    recall_at_5: float = 0.0
    mrr_at_3: float = 0.0
    mrr_at_5: float = 0.0

    # Schema-level metrics (full API match)
    schema_recall_at_1: float = 0.0
    schema_recall_at_2: float = 0.0
    schema_recall_at_3: float = 0.0
    schema_recall_at_4: float = 0.0
    schema_recall_at_5: float = 0.0
    schema_mrr_at_3: float = 0.0
    schema_mrr_at_5: float = 0.0

    # Sample count
    num_samples: int = 0

    def to_dict(self) -> Dict:
        return {
            "service_level": {
                "R@1": round(self.recall_at_1, 4),
                "R@2": round(self.recall_at_2, 4),
                "R@3": round(self.recall_at_3, 4),
                "R@4": round(self.recall_at_4, 4),
                "R@5": round(self.recall_at_5, 4),
                "MRR@3": round(self.mrr_at_3, 4),
                "MRR@5": round(self.mrr_at_5, 4),
            },
            "schema_level": {
                "R@1": round(self.schema_recall_at_1, 4),
                "R@2": round(self.schema_recall_at_2, 4),
                "R@3": round(self.schema_recall_at_3, 4),
                "R@4": round(self.schema_recall_at_4, 4),
                "R@5": round(self.schema_recall_at_5, 4),
                "MRR@3": round(self.schema_mrr_at_3, 4),
                "MRR@5": round(self.schema_mrr_at_5, 4),
            },
            "num_samples": self.num_samples,
        }


@dataclass
class JointMetrics:
    """Joint metrics (both trigger AND action correct)."""
    # Service-level
    joint_recall_at_1: float = 0.0
    joint_recall_at_2: float = 0.0
    joint_recall_at_3: float = 0.0
    joint_recall_at_4: float = 0.0
    joint_recall_at_5: float = 0.0

    # Schema-level
    schema_joint_recall_at_1: float = 0.0
    schema_joint_recall_at_2: float = 0.0
    schema_joint_recall_at_3: float = 0.0
    schema_joint_recall_at_4: float = 0.0
    schema_joint_recall_at_5: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "service_level": {
                "R@1": round(self.joint_recall_at_1, 4),
                "R@2": round(self.joint_recall_at_2, 4),
                "R@3": round(self.joint_recall_at_3, 4),
                "R@4": round(self.joint_recall_at_4, 4),
                "R@5": round(self.joint_recall_at_5, 4),
            },
            "schema_level": {
                "R@1": round(self.schema_joint_recall_at_1, 4),
                "R@2": round(self.schema_joint_recall_at_2, 4),
                "R@3": round(self.schema_joint_recall_at_3, 4),
                "R@4": round(self.schema_joint_recall_at_4, 4),
                "R@5": round(self.schema_joint_recall_at_5, 4),
            },
        }


@dataclass
class AblationResult:
    """Comprehensive results from a single ablation run."""
    method: str
    config: Dict[str, Any]

    # Comprehensive retrieval metrics
    trigger_metrics: Dict = field(default_factory=dict)
    action_metrics: Dict = field(default_factory=dict)
    joint_metrics: Dict = field(default_factory=dict)

    # Legacy fields for backward compatibility (service-level R@5)
    trigger_accuracy: float = 0.0  # R@5 service-level
    action_accuracy: float = 0.0   # R@5 service-level
    joint_accuracy: float = 0.0    # R@5 service-level

    # Training stats
    trainable_params: int = 0
    total_params: int = 0
    trainable_pct: float = 0.0
    training_time_seconds: float = 0.0

    # Timestamp
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


def load_json(filepath: Path) -> List[Dict]:
    """Load JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


# =============================================================================
# COMPREHENSIVE RETRIEVAL EVALUATION
# =============================================================================

def find_rank_service(target_name: str, corpus_names: List[str], top_indices: List[int], max_k: int = 10) -> int:
    """
    Find rank matching service_name only.
    Returns max_k+1 if not found.
    """
    target_norm = normalize_name(target_name)
    for rank, idx in enumerate(top_indices[:max_k], 1):
        if normalize_name(corpus_names[idx]) == target_norm:
            return rank
    return max_k + 1


def find_rank_schema(
    target_name: str,
    target_api_info: Dict,
    corpus_names: List[str],
    corpus_api_infos: List[Dict],
    top_indices: List[int],
    is_trigger: bool = True,
    max_k: int = 10
) -> int:
    """
    Find rank matching BOTH service_name AND schema (ingredients/fields).
    Returns max_k+1 if not found.
    """
    target_norm = normalize_name(target_name)
    target_schema = get_schema_key(target_api_info, is_trigger)

    for rank, idx in enumerate(top_indices[:max_k], 1):
        if normalize_name(corpus_names[idx]) == target_norm:
            result_schema = get_schema_key(corpus_api_infos[idx], is_trigger)
            if result_schema == target_schema:
                return rank
    return max_k + 1


def evaluate_retrieval_comprehensive(
    trigger_model: SentenceTransformer,
    action_model: SentenceTransformer,
    test_path: Path,
    triggers_path: Path,
    actions_path: Path,
    max_k: int = 10
) -> Tuple[RetrievalMetrics, RetrievalMetrics, JointMetrics]:
    """
    Comprehensive retrieval evaluation with all metrics.

    Computes:
    - Service-level: R@1, R@2, R@3, R@4, R@5, MRR@3, MRR@5
    - Schema-level: Same metrics but with full API matching

    Returns:
        Tuple of (trigger_metrics, action_metrics, joint_metrics)
    """
    from rag.indexer import extract_trigger_text, extract_action_text

    # Load data
    test_applets = load_json(test_path)
    triggers_corpus = load_json(triggers_path)
    actions_corpus = load_json(actions_path)

    # Build corpus data
    trigger_texts = []
    trigger_names = []
    trigger_api_infos = []
    for t in triggers_corpus:
        text = extract_trigger_text(t)
        if text:
            trigger_texts.append(text)
            trigger_names.append(t.get("service_name", ""))
            trigger_api_infos.append(t.get("api_info", {}))

    action_texts = []
    action_names = []
    action_api_infos = []
    for a in actions_corpus:
        text = extract_action_text(a)
        if text:
            action_texts.append(text)
            action_names.append(a.get("service_name", ""))
            action_api_infos.append(a.get("api_info", {}))

    # Encode corpus
    print(f"  Encoding {len(trigger_texts)} triggers and {len(action_texts)} actions...")
    trigger_embeddings = trigger_model.encode(trigger_texts, convert_to_tensor=True, show_progress_bar=False)
    action_embeddings = action_model.encode(action_texts, convert_to_tensor=True, show_progress_bar=False)

    # Collect ranks for all samples
    trigger_service_ranks = []
    trigger_schema_ranks = []
    action_service_ranks = []
    action_schema_ranks = []

    # Joint tracking (both correct at each k)
    joint_service_at_k = {k: 0 for k in [1, 2, 3, 4, 5]}
    joint_schema_at_k = {k: 0 for k in [1, 2, 3, 4, 5]}

    total = 0

    for applet in test_applets:
        query = applet.get("query", "")
        if not query or len(query) < 10:
            continue

        # Get ground truth
        gt_trigger_name = applet.get("trigger", {}).get("service_name", "")
        gt_trigger_api = applet.get("trigger", {}).get("api_info", {})
        gt_action_name = applet.get("action", {}).get("service_name", "")
        gt_action_api = applet.get("action", {}).get("api_info", {})

        if not gt_trigger_name or not gt_action_name:
            continue

        total += 1

        # Encode query
        q_trig = trigger_model.encode(query, convert_to_tensor=True, show_progress_bar=False)
        q_act = action_model.encode(query, convert_to_tensor=True, show_progress_bar=False)

        # Find top-k triggers
        trig_sims = torch.nn.functional.cosine_similarity(q_trig.unsqueeze(0), trigger_embeddings)
        top_trig_idx = torch.topk(trig_sims, min(max_k, len(trigger_names))).indices.tolist()

        # Find top-k actions
        act_sims = torch.nn.functional.cosine_similarity(q_act.unsqueeze(0), action_embeddings)
        top_act_idx = torch.topk(act_sims, min(max_k, len(action_names))).indices.tolist()

        # Compute ranks - Service level
        trig_service_rank = find_rank_service(gt_trigger_name, trigger_names, top_trig_idx, max_k)
        act_service_rank = find_rank_service(gt_action_name, action_names, top_act_idx, max_k)
        trigger_service_ranks.append(trig_service_rank)
        action_service_ranks.append(act_service_rank)

        # Compute ranks - Schema level
        trig_schema_rank = find_rank_schema(
            gt_trigger_name, gt_trigger_api, trigger_names, trigger_api_infos,
            top_trig_idx, is_trigger=True, max_k=max_k
        )
        act_schema_rank = find_rank_schema(
            gt_action_name, gt_action_api, action_names, action_api_infos,
            top_act_idx, is_trigger=False, max_k=max_k
        )
        trigger_schema_ranks.append(trig_schema_rank)
        action_schema_ranks.append(act_schema_rank)

        # Joint accuracy at each k
        for k in [1, 2, 3, 4, 5]:
            if trig_service_rank <= k and act_service_rank <= k:
                joint_service_at_k[k] += 1
            if trig_schema_rank <= k and act_schema_rank <= k:
                joint_schema_at_k[k] += 1

    # Compute trigger metrics
    trigger_metrics = RetrievalMetrics(
        recall_at_1=compute_recall_at_k(trigger_service_ranks, 1),
        recall_at_2=compute_recall_at_k(trigger_service_ranks, 2),
        recall_at_3=compute_recall_at_k(trigger_service_ranks, 3),
        recall_at_4=compute_recall_at_k(trigger_service_ranks, 4),
        recall_at_5=compute_recall_at_k(trigger_service_ranks, 5),
        mrr_at_3=compute_mrr_at_k(trigger_service_ranks, 3),
        mrr_at_5=compute_mrr_at_k(trigger_service_ranks, 5),
        schema_recall_at_1=compute_recall_at_k(trigger_schema_ranks, 1),
        schema_recall_at_2=compute_recall_at_k(trigger_schema_ranks, 2),
        schema_recall_at_3=compute_recall_at_k(trigger_schema_ranks, 3),
        schema_recall_at_4=compute_recall_at_k(trigger_schema_ranks, 4),
        schema_recall_at_5=compute_recall_at_k(trigger_schema_ranks, 5),
        schema_mrr_at_3=compute_mrr_at_k(trigger_schema_ranks, 3),
        schema_mrr_at_5=compute_mrr_at_k(trigger_schema_ranks, 5),
        num_samples=total,
    )

    # Compute action metrics
    action_metrics = RetrievalMetrics(
        recall_at_1=compute_recall_at_k(action_service_ranks, 1),
        recall_at_2=compute_recall_at_k(action_service_ranks, 2),
        recall_at_3=compute_recall_at_k(action_service_ranks, 3),
        recall_at_4=compute_recall_at_k(action_service_ranks, 4),
        recall_at_5=compute_recall_at_k(action_service_ranks, 5),
        mrr_at_3=compute_mrr_at_k(action_service_ranks, 3),
        mrr_at_5=compute_mrr_at_k(action_service_ranks, 5),
        schema_recall_at_1=compute_recall_at_k(action_schema_ranks, 1),
        schema_recall_at_2=compute_recall_at_k(action_schema_ranks, 2),
        schema_recall_at_3=compute_recall_at_k(action_schema_ranks, 3),
        schema_recall_at_4=compute_recall_at_k(action_schema_ranks, 4),
        schema_recall_at_5=compute_recall_at_k(action_schema_ranks, 5),
        schema_mrr_at_3=compute_mrr_at_k(action_schema_ranks, 3),
        schema_mrr_at_5=compute_mrr_at_k(action_schema_ranks, 5),
        num_samples=total,
    )

    # Compute joint metrics
    joint_metrics = JointMetrics(
        joint_recall_at_1=joint_service_at_k[1] / total if total > 0 else 0,
        joint_recall_at_2=joint_service_at_k[2] / total if total > 0 else 0,
        joint_recall_at_3=joint_service_at_k[3] / total if total > 0 else 0,
        joint_recall_at_4=joint_service_at_k[4] / total if total > 0 else 0,
        joint_recall_at_5=joint_service_at_k[5] / total if total > 0 else 0,
        schema_joint_recall_at_1=joint_schema_at_k[1] / total if total > 0 else 0,
        schema_joint_recall_at_2=joint_schema_at_k[2] / total if total > 0 else 0,
        schema_joint_recall_at_3=joint_schema_at_k[3] / total if total > 0 else 0,
        schema_joint_recall_at_4=joint_schema_at_k[4] / total if total > 0 else 0,
        schema_joint_recall_at_5=joint_schema_at_k[5] / total if total > 0 else 0,
    )

    return trigger_metrics, action_metrics, joint_metrics


# =============================================================================
# MODEL CREATION AND TRAINING
# =============================================================================

def apply_lora(model: SentenceTransformer, rank: int, alpha: int = None) -> SentenceTransformer:
    """Apply LoRA to sentence transformer model."""
    if not PEFT_AVAILABLE:
        raise ImportError("PEFT not installed. Run: pip install peft")

    alpha = alpha or rank * 2

    # Get the base transformer model
    base_model = model[0].auto_model

    # Configure LoRA
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Attention layers
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )

    # Apply LoRA
    peft_model = get_peft_model(base_model, lora_config)

    # Replace the base model
    model[0].auto_model = peft_model

    return model


def apply_layer_freezing(model: SentenceTransformer, freeze_layers: int = 12) -> SentenceTransformer:
    """Apply layer freezing to model."""
    base_model = model[0].auto_model

    # Freeze embedding layer
    for param in base_model.embed_tokens.parameters():
        param.requires_grad = False

    # Freeze specified number of layers
    for i in range(freeze_layers):
        if i < len(base_model.layers):
            for param in base_model.layers[i].parameters():
                param.requires_grad = False

    return model


def count_parameters(model: SentenceTransformer) -> Tuple[int, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def create_model_with_method(method: str, lora_rank: Optional[int] = None, freeze_layers: Optional[int] = None):
    """Create a model with the specified adaptation method."""
    model = SentenceTransformer(
        config.base_model,
        model_kwargs={"torch_dtype": torch.float32}
    )

    if method == "lora":
        model = apply_lora(model, rank=lora_rank)
    elif method == "layer_freeze":
        model = apply_layer_freezing(model, freeze_layers=freeze_layers)
    # else: full_finetune - no modifications

    return model


def train_single_encoder(
    model: SentenceTransformer,
    train_dataset: Dataset,
    val_dataset: Dataset,
    output_dir: Path,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
) -> float:
    """Train a single encoder and return training time."""

    # Loss function
    loss = losses.MultipleNegativesRankingLoss(
        model=model,
        scale=1.0 / 0.05,  # temperature = 0.05
    )

    # Training arguments
    args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=False,
        bf16=False,
        logging_steps=50,
        save_strategy="epoch",  # Save checkpoints each epoch
        save_total_limit=1,  # Keep only the last checkpoint
        eval_strategy="epoch",
        report_to="none",
        seed=42,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss=loss,
    )

    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time

    # Save final model
    final_path = output_dir / "final"
    final_path.mkdir(parents=True, exist_ok=True)
    model.save(str(final_path))
    print(f"Model saved to: {final_path}")

    return training_time


# =============================================================================
# MAIN ABLATION RUNNER
# =============================================================================

def run_ablation(
    method: str,
    lora_rank: Optional[int] = None,
    freeze_layers: Optional[int] = None,
    epochs: int = 3,
    batch_size: int = 16,
) -> AblationResult:
    """
    Run a single ablation experiment with BOTH trigger and action encoders.

    Args:
        method: One of "full_finetune", "lora", "layer_freeze"
        lora_rank: LoRA rank (if method == "lora")
        freeze_layers: Number of layers to freeze (if method == "layer_freeze")
    """
    print(f"\n{'='*60}")
    print(f"Running ablation: {method}" + (f" (r={lora_rank})" if lora_rank else ""))
    print(f"{'='*60}")

    # Determine method name
    if method == "lora":
        method_name = f"lora_r{lora_rank}"
    elif method == "layer_freeze":
        method_name = "layer_freeze"
    else:
        method_name = "full_finetune"

    total_training_time = 0

    # =========================================================================
    # TRAIN TRIGGER ENCODER
    # =========================================================================
    print("\n--- Training TRIGGER Encoder ---")
    trigger_model = create_model_with_method(method, lora_rank, freeze_layers)

    # Count parameters (same for both models)
    total_params, trainable_params = count_parameters(trigger_model)
    trainable_pct = trainable_params / total_params * 100
    print(f"Parameters: {trainable_params:,} trainable / {total_params:,} total ({trainable_pct:.1f}%)")

    # Build trigger training data
    print("Building trigger training pairs...")
    trigger_train, trigger_val = build_trigger_pairs(
        config.applets_path,
        val_split=0.1,
        seed=42
    )

    # Train trigger encoder
    trigger_output_dir = config.project_root / "models" / f"ablation_{method_name}_trigger"
    print(f"Training trigger encoder for {epochs} epochs...")
    trigger_time = train_single_encoder(
        model=trigger_model,
        train_dataset=trigger_train,
        val_dataset=trigger_val,
        output_dir=trigger_output_dir,
        epochs=epochs,
        batch_size=batch_size,
    )
    total_training_time += trigger_time
    print(f"Trigger encoder training completed in {trigger_time/60:.2f} minutes")

    # =========================================================================
    # TRAIN ACTION ENCODER
    # =========================================================================
    print("\n--- Training ACTION Encoder ---")
    action_model = create_model_with_method(method, lora_rank, freeze_layers)

    # Build action training data
    print("Building action training pairs...")
    action_train, action_val = build_action_pairs(
        config.applets_path,
        val_split=0.1,
        seed=42
    )

    # Train action encoder
    action_output_dir = config.project_root / "models" / f"ablation_{method_name}_action"
    print(f"Training action encoder for {epochs} epochs...")
    action_time = train_single_encoder(
        model=action_model,
        train_dataset=action_train,
        val_dataset=action_val,
        output_dir=action_output_dir,
        epochs=epochs,
        batch_size=batch_size,
    )
    total_training_time += action_time
    print(f"Action encoder training completed in {action_time/60:.2f} minutes")

    # =========================================================================
    # COMPREHENSIVE EVALUATION
    # =========================================================================
    print("\n--- Comprehensive Evaluation ---")
    print("Computing R@1, R@2, R@3, R@4, R@5, MRR@3, MRR@5 (service + schema level)...")

    trigger_metrics, action_metrics, joint_metrics = evaluate_retrieval_comprehensive(
        trigger_model=trigger_model,
        action_model=action_model,
        test_path=config.project_root / "data" / "test" / "gold.json",
        triggers_path=config.triggers_rag_path,
        actions_path=config.actions_rag_path,
        max_k=10
    )

    # Print summary
    print("\n  TRIGGER Metrics (Service-Level):")
    print(f"    R@1={trigger_metrics.recall_at_1:.4f}, R@3={trigger_metrics.recall_at_3:.4f}, R@5={trigger_metrics.recall_at_5:.4f}")
    print(f"    MRR@3={trigger_metrics.mrr_at_3:.4f}, MRR@5={trigger_metrics.mrr_at_5:.4f}")
    print("  TRIGGER Metrics (Schema-Level):")
    print(f"    R@1={trigger_metrics.schema_recall_at_1:.4f}, R@3={trigger_metrics.schema_recall_at_3:.4f}, R@5={trigger_metrics.schema_recall_at_5:.4f}")

    print("\n  ACTION Metrics (Service-Level):")
    print(f"    R@1={action_metrics.recall_at_1:.4f}, R@3={action_metrics.recall_at_3:.4f}, R@5={action_metrics.recall_at_5:.4f}")
    print(f"    MRR@3={action_metrics.mrr_at_3:.4f}, MRR@5={action_metrics.mrr_at_5:.4f}")
    print("  ACTION Metrics (Schema-Level):")
    print(f"    R@1={action_metrics.schema_recall_at_1:.4f}, R@3={action_metrics.schema_recall_at_3:.4f}, R@5={action_metrics.schema_recall_at_5:.4f}")

    print("\n  JOINT Metrics (Both Correct):")
    print(f"    Service: R@1={joint_metrics.joint_recall_at_1:.4f}, R@3={joint_metrics.joint_recall_at_3:.4f}, R@5={joint_metrics.joint_recall_at_5:.4f}")
    print(f"    Schema:  R@1={joint_metrics.schema_joint_recall_at_1:.4f}, R@3={joint_metrics.schema_joint_recall_at_3:.4f}, R@5={joint_metrics.schema_joint_recall_at_5:.4f}")

    # Build result
    result = AblationResult(
        method=method_name,
        config={
            "lora_rank": lora_rank,
            "freeze_layers": freeze_layers,
            "epochs": epochs,
            "batch_size": batch_size,
        },
        trigger_metrics=trigger_metrics.to_dict(),
        action_metrics=action_metrics.to_dict(),
        joint_metrics=joint_metrics.to_dict(),
        # Legacy fields (service-level R@5 as percentage)
        trigger_accuracy=trigger_metrics.recall_at_5 * 100,
        action_accuracy=action_metrics.recall_at_5 * 100,
        joint_accuracy=joint_metrics.joint_recall_at_5 * 100,
        trainable_params=trainable_params,
        total_params=total_params,
        trainable_pct=trainable_pct,
        training_time_seconds=total_training_time,
    )

    # Cleanup both models
    del trigger_model
    del action_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def run_full_ablation_study(
    lora_ranks: List[int] = [8, 16, 32],
    freeze_layers: int = 12,
    epochs: int = 3,
    batch_size: int = 16,
    output_path: Path = None,
) -> Dict[str, Any]:
    """
    Run complete ablation study comparing all methods.

    Args:
        lora_ranks: List of LoRA ranks to test
        freeze_layers: Number of layers to freeze for layer freezing method
        epochs: Training epochs per method
        batch_size: Training batch size
        output_path: Path to save JSON results

    Returns:
        Dictionary with all results
    """
    output_path = output_path or (config.project_root / "results" / "ablation_comprehensive.json")

    print("\n" + "="*70)
    print("FARM COMPREHENSIVE RETRIEVAL ABLATION STUDY")
    print("="*70)
    print(f"Methods to test:")
    print(f"  - Full fine-tuning")
    for r in lora_ranks:
        print(f"  - LoRA (r={r})")
    print(f"  - Layer freezing ({freeze_layers} layers frozen)")
    print(f"\nMetrics computed:")
    print(f"  - Service-level: R@1, R@2, R@3, R@4, R@5, MRR@3, MRR@5")
    print(f"  - Schema-level:  R@1, R@2, R@3, R@4, R@5, MRR@3, MRR@5")
    print(f"  - Joint (both):  R@1, R@2, R@3, R@4, R@5")
    print(f"\nEpochs per method: {epochs}")
    print(f"Output: {output_path}")
    print("="*70)

    results = {
        "study_info": {
            "timestamp": datetime.now().isoformat(),
            "base_model": config.base_model,
            "lora_ranks_tested": lora_ranks,
            "freeze_layers": freeze_layers,
            "epochs": epochs,
            "batch_size": batch_size,
            "metrics_computed": {
                "service_level": ["R@1", "R@2", "R@3", "R@4", "R@5", "MRR@3", "MRR@5"],
                "schema_level": ["R@1", "R@2", "R@3", "R@4", "R@5", "MRR@3", "MRR@5"],
                "joint": ["R@1", "R@2", "R@3", "R@4", "R@5"],
            }
        },
        "results": []
    }

    total_methods = 2 + len(lora_ranks)

    # 1. Full fine-tuning
    print(f"\n[1/{total_methods}] Full Fine-tuning")
    result = run_ablation(
        method="full_finetune",
        epochs=epochs,
        batch_size=batch_size,
    )
    results["results"].append(asdict(result))
    _save_intermediate(results, output_path)

    # 2. LoRA variants
    for i, rank in enumerate(lora_ranks):
        print(f"\n[{i+2}/{total_methods}] LoRA (r={rank})")
        result = run_ablation(
            method="lora",
            lora_rank=rank,
            epochs=epochs,
            batch_size=batch_size,
        )
        results["results"].append(asdict(result))
        _save_intermediate(results, output_path)

    # 3. Layer freezing (our method)
    print(f"\n[{total_methods}/{total_methods}] Layer Freezing")
    result = run_ablation(
        method="layer_freeze",
        freeze_layers=freeze_layers,
        epochs=epochs,
        batch_size=batch_size,
    )
    results["results"].append(asdict(result))

    # Save final results
    _save_results(results, output_path)

    # Print summary
    _print_summary(results)

    return results


def _save_intermediate(results: Dict, path: Path):
    """Save intermediate results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)


def _save_results(results: Dict, path: Path):
    """Save final results to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {path}")


def _print_summary(results: Dict):
    """Print comprehensive summary table of results."""
    print("\n" + "="*100)
    print("COMPREHENSIVE ABLATION STUDY RESULTS")
    print("="*100)

    # Service-level table
    print("\n--- SERVICE-LEVEL METRICS ---")
    print(f"{'Method':<15} {'Trig R@1':>8} {'Trig R@5':>8} {'Act R@1':>8} {'Act R@5':>8} {'Joint R@1':>9} {'Joint R@5':>9} {'Params':>14}")
    print("-"*100)

    for r in results["results"]:
        params_str = f"{r['trainable_params']/1e6:.1f}M ({r['trainable_pct']:.1f}%)"
        trig = r['trigger_metrics']['service_level']
        act = r['action_metrics']['service_level']
        joint = r['joint_metrics']['service_level']
        print(f"{r['method']:<15} {trig['R@1']:>8.4f} {trig['R@5']:>8.4f} "
              f"{act['R@1']:>8.4f} {act['R@5']:>8.4f} "
              f"{joint['R@1']:>9.4f} {joint['R@5']:>9.4f} {params_str:>14}")

    # Schema-level table
    print("\n--- SCHEMA-LEVEL METRICS (Full API Match) ---")
    print(f"{'Method':<15} {'Trig R@1':>8} {'Trig R@5':>8} {'Act R@1':>8} {'Act R@5':>8} {'Joint R@1':>9} {'Joint R@5':>9}")
    print("-"*90)

    for r in results["results"]:
        trig = r['trigger_metrics']['schema_level']
        act = r['action_metrics']['schema_level']
        joint = r['joint_metrics']['schema_level']
        print(f"{r['method']:<15} {trig['R@1']:>8.4f} {trig['R@5']:>8.4f} "
              f"{act['R@1']:>8.4f} {act['R@5']:>8.4f} "
              f"{joint['R@1']:>9.4f} {joint['R@5']:>9.4f}")

    # MRR table
    print("\n--- MRR METRICS (Service-Level) ---")
    print(f"{'Method':<15} {'Trig MRR@3':>10} {'Trig MRR@5':>10} {'Act MRR@3':>10} {'Act MRR@5':>10}")
    print("-"*60)

    for r in results["results"]:
        trig = r['trigger_metrics']['service_level']
        act = r['action_metrics']['service_level']
        print(f"{r['method']:<15} {trig['MRR@3']:>10.4f} {trig['MRR@5']:>10.4f} "
              f"{act['MRR@3']:>10.4f} {act['MRR@5']:>10.4f}")

    print("\n" + "="*100)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run comprehensive retrieval ablation study")
    parser.add_argument("--ranks", type=int, nargs="+", default=[8, 16, 32],
                        help="LoRA ranks to test (default: 8 16 32)")
    parser.add_argument("--freeze-layers", type=int, default=12,
                        help="Layers to freeze for layer freezing method (default: 12)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Training epochs per method (default: 3)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Training batch size (default: 16)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: results/ablation_comprehensive.json)")

    args = parser.parse_args()

    output_path = Path(args.output) if args.output else None

    results = run_full_ablation_study(
        lora_ranks=args.ranks,
        freeze_layers=args.freeze_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_path=output_path,
    )
