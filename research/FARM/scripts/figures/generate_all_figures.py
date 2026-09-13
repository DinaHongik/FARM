#!/usr/bin/env python3
"""
FARM Paper Figures - Q1 Journal Style
=====================================
Generates all figures for the experiments section.

STAGE 1 - CONTRASTIVE LEARNING:
    Fig 4: Encoder Retrieval Performance (Baseline vs Contrastive-Trained)
    Fig 5: Contrastive Training Convergence
    Fig 6: Retrieval Quality (Context Recall & Precision)

STAGE 2 - MULTI-AGENT SELECTION:
    Fig 7: Selection Performance (Goal, Trigger, Action, Joint Accuracy)
    Fig 8: Quality Metrics (Faithfulness, Topic Adherence)

ABLATION STUDIES:
    Fig 9: Layer Freezing Ablation
    Fig 10: Component Contribution

Usage:
    python scripts/figures/generate_all_figures.py

Output:
    latex/figures/fig4_encoder_performance.pdf
    latex/figures/fig5_training_curves.pdf
    latex/figures/fig6_retrieval_quality.pdf
    latex/figures/fig7_selection_performance.pdf
    latex/figures/fig8_quality_metrics.pdf
    latex/figures/fig9_layer_freezing.pdf
    latex/figures/fig10_component_ablation.pdf
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# ============================================================
# SECTION 1: Q1 JOURNAL STYLE CONFIGURATION
# ============================================================

def setup_journal_style():
    """Configure matplotlib for Q1 journal publication quality (grayscale-compatible)."""
    # Use a clean white background style
    plt.style.use('seaborn-v0_8-whitegrid')

    mpl.rcParams.update({
        # Font settings - serif for academic publications (Times-like)
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Palatino', 'Georgia'],
        'font.size': 9,

        # Figure settings - high resolution for print
        'figure.dpi': 300,
        'figure.facecolor': 'white',
        'savefig.dpi': 300,
        'savefig.format': 'pdf',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'savefig.facecolor': 'white',

        # Axes settings - clean academic look
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'axes.titleweight': 'bold',
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'axes.axisbelow': True,
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',

        # Grid settings - subtle gray grid
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'grid.color': '#cccccc',

        # Tick settings
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.color': 'black',
        'ytick.color': 'black',

        # Legend settings
        'legend.fontsize': 9,
        'legend.framealpha': 1.0,
        'legend.edgecolor': 'black',
        'legend.facecolor': 'white',

        # Line settings
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'lines.color': 'black',

        # Hatch settings - black hatches for grayscale visibility
        'hatch.color': 'black',
        'hatch.linewidth': 0.8,

        # Patch (bar) settings
        'patch.edgecolor': 'black',
        'patch.linewidth': 1.0,

        # Text color
        'text.color': 'black',
        'axes.labelcolor': 'black',
    })

# =============================================================================
# Q1 JOURNAL GRAYSCALE STYLE - IEEE/ACM Compatible
# =============================================================================
# For grayscale printing: Use WHITE fills with distinct BLACK hatch patterns
# This ensures figures are readable when printed in black & white
# Reference: IEEE requires figures to be readable in grayscale

# Subtle, muted color palette - professional Q1 journal style
# Light pastel colors that are distinguishable in both color AND grayscale
COLORS = {
    'baseline': '#f0f0f0',      # Light gray - solid (no hatch)
    'trained': '#cce5ff',       # Light blue - with diagonal hatch
    'trigger': '#d4edda',       # Light green - with backward diagonal
    'action': '#fff3cd',        # Light yellow/cream - with cross-hatch
    'gold': '#cce5ff',          # Light blue - with diagonal hatch
    'noisy': '#f8d7da',         # Light pink/salmon - with horizontal hatch
    'oneshot': '#d4edda',       # Light green - with cross pattern
    'highlight': '#e2e3e5',     # Medium gray for emphasis
    'secondary': '#e2d5f1',     # Light lavender - with dots
    'recall': '#cce5ff',        # Light blue
    'precision': '#fff3cd',     # Light cream
    'faithfulness': '#d4edda',  # Light green
    'adherence': '#e2d5f1',     # Light lavender
}

# Dense, distinct hatch patterns for clear grayscale differentiation
# Pattern density increased (more characters = denser pattern)
HATCHES = {
    'baseline': '',              # Solid (no hatch) - clearly different
    'trained': '////',           # Dense diagonal lines
    'trigger': '\\\\\\\\',       # Dense backward diagonal
    'action': 'xxxx',            # Dense cross-hatch
    'gold': '////',              # Dense diagonal lines
    'noisy': '----',             # Horizontal lines
    'oneshot': 'xxxx',           # Dense cross pattern
    'highlight': '\\\\\\\\',     # Backward diagonal
    'secondary': '....',         # Dense dots
}

# Figure sizes (inches) - single column ~3.5", double column ~7"
FIG_SINGLE = (3.5, 2.8)
FIG_DOUBLE = (7.0, 3.0)
FIG_DOUBLE_TALL = (7.0, 3.5)

# ============================================================
# SECTION 2: HARD-CODED DATA (from README.md)
# ============================================================

# Service-Level Metrics - Test Gold
ENCODER_GOLD = {
    'metrics': ['R@1', 'R@3', 'R@5', 'MRR@3'],
    'trigger': {
        'baseline': [0.30, 0.52, 0.59, 0.39],
        'trained': [0.74, 0.85, 0.92, 0.79],
    },
    'action': {
        'baseline': [0.39, 0.53, 0.57, 0.45],
        'trained': [0.74, 0.88, 0.91, 0.80],
    },
    'combined_r1': {
        'baseline': 0.07,
        'trained': 0.55,
    }
}

# Service-Level Metrics - Test Noisy
ENCODER_NOISY = {
    'metrics': ['R@1', 'R@3', 'R@5', 'MRR@3'],
    'trigger': {
        'baseline': [0.21, 0.39, 0.47, 0.29],
        'trained': [0.63, 0.77, 0.82, 0.69],
    },
    'action': {
        'baseline': [0.20, 0.36, 0.42, 0.27],
        'trained': [0.47, 0.73, 0.78, 0.58],
    },
    'combined_r1': {
        'baseline': 0.05,
        'trained': 0.30,
    }
}

# Service-Level Metrics - Test One-shot
ENCODER_ONESHOT = {
    'metrics': ['R@1', 'R@3', 'R@5', 'MRR@3'],
    'trigger': {
        'baseline': [0.25, 0.41, 0.46, 0.32],
        'trained': [0.66, 0.82, 0.86, 0.73],
    },
    'action': {
        'baseline': [0.34, 0.48, 0.53, 0.41],
        'trained': [0.69, 0.82, 0.86, 0.75],
    },
    'combined_r1': {
        'baseline': 0.11,
        'trained': 0.48,
    }
}

# Training Loss Curves (from README.md training logs)
TRAINING_LOSS = {
    'trigger': {
        'epochs': [0.13, 0.51, 1.0, 1.52, 2.0, 2.4, 3.0],
        'train_loss': [1.1063, 0.4989, None, 0.2271, None, 0.1423, None],
        'eval_loss': [None, None, 0.3371, None, 0.2875, None, 0.2621],
    },
    'action': {
        'epochs': [0.13, 0.51, 1.0, 1.52, 2.0, 2.53, 3.0],
        'train_loss': [1.2842, 0.6544, None, 0.3449, None, 0.1992, None],
        'eval_loss': [None, None, 0.5076, None, 0.4426, None, 0.4343],
    }
}

# Layer Freezing Ablation - Using R@5 for consistency with LoRA ablation (Fig 11)
# Data from actual LoRA ablation experiments (train_lora_ablation.py with k=5)
LAYER_FREEZING = {
    'configurations': ['Full\nContrastive', 'Layer\nFreeze'],
    'trigger_r5': [0.92, 0.94],  # From LORA_ABLATION
    'action_r5': [0.90, 0.91],
    'joint_r5': [0.82, 0.86],
    'trainable_pct': [100.0, 18.0],
}

# Component Ablation (incremental contributions)
COMPONENT_ABLATION = {
    'components': ['Pretrained\nBaseline', '+ Contrastive\n(Full)', '+ Layer\nFreezing', '+ Multi-Agent\nSelection'],
    'trigger_r1': [0.30, 0.74, 0.74, 0.89],  # Last from eval.json gold
    'action_r1': [0.39, 0.74, 0.74, 0.90],
    'joint_accuracy': [0.07, 0.55, 0.55, 0.81],
}

# LoRA Ablation Study - Comparing adaptation methods (from actual experiments)
# Shows: Layer Freezing > Full Fine-tuning > LoRA for TAP domain
LORA_ABLATION = {
    'methods': ['Full\nFine-tune', 'LoRA\n(r=8)', 'LoRA\n(r=16)', 'LoRA\n(r=32)', 'Layer\nFreeze'],
    'trigger_r5': [0.92, 0.85, 0.85, 0.85, 0.94],
    'action_r5': [0.90, 0.83, 0.85, 0.87, 0.91],
    'joint_r5': [0.82, 0.69, 0.71, 0.73, 0.86],
    'trainable_pct': [100.0, 1.8, 2.2, 2.8, 18.0],
}

# ============================================================
# SECTION 3: LOAD AGENTIC DATA (from eval.json)
# ============================================================

def load_eval_json(path: str = 'results/eval.json') -> dict:
    """Load evaluation results from eval.json."""
    # Try relative to script location, then relative to CWD
    script_dir = Path(__file__).parent.parent.parent
    eval_path = script_dir / path

    if not eval_path.exists():
        eval_path = Path(path)

    if not eval_path.exists():
        print(f"Warning: {path} not found. Using placeholder data for agentic figures.")
        return None

    with open(eval_path, 'r') as f:
        return json.load(f)


def extract_agentic_metrics(eval_data: dict) -> dict:
    """Extract key metrics from eval.json for figures."""
    if eval_data is None:
        # Placeholder data from actual eval.json
        return {
            'gold': {
                'goal_acc': 0.895, 'trigger_acc': 0.89, 'action_acc': 0.90,
                'joint_acc': 0.81, 'success_rate': 0.98, 'faithfulness': 0.496,
                'topic_adherence': 0.819,
                'context_recall_trigger': 0.96, 'context_recall_action': 0.95,
                'context_precision_trigger': 0.784, 'context_precision_action': 0.779,
            },
            'noisy': {
                'goal_acc': 0.765, 'trigger_acc': 0.83, 'action_acc': 0.70,
                'joint_acc': 0.62, 'success_rate': 0.91, 'faithfulness': 0.450,
                'topic_adherence': 0.817,
                'context_recall_trigger': 0.89, 'context_recall_action': 0.89,
                'context_precision_trigger': 0.692, 'context_precision_action': 0.669,
            },
            'oneshot': {
                'goal_acc': 0.83, 'trigger_acc': 0.80, 'action_acc': 0.86,
                'joint_acc': 0.71, 'success_rate': 0.95, 'faithfulness': 0.440,
                'topic_adherence': 0.812,
                'context_recall_trigger': 0.94, 'context_recall_action': 0.95,
                'context_precision_trigger': 0.771, 'context_precision_action': 0.753,
            },
        }

    result = {}
    for split in ['gold', 'noisy', 'oneshot']:
        metrics = eval_data['summary'][split]['metrics']
        result[split] = {
            'goal_acc': metrics['goal_accuracy'],
            'trigger_acc': metrics['trigger_accuracy'],
            'action_acc': metrics['action_accuracy'],
            'joint_acc': metrics['joint_accuracy'],
            'success_rate': eval_data['summary'][split]['success_rate'],
            'faithfulness': metrics['faithfulness'],
            'topic_adherence': metrics['topic_adherence'],
            'context_recall_trigger': metrics['context_recall_trigger'],
            'context_recall_action': metrics['context_recall_action'],
            'context_precision_trigger': metrics['context_precision_trigger'],
            'context_precision_action': metrics['context_precision_action'],
        }
    return result


def extract_verifier_scores(eval_data: dict) -> dict:
    """Extract verifier scores for distribution plot."""
    if eval_data is None:
        return {'gold': [], 'noisy': [], 'oneshot': []}

    scores = {}
    for split in ['gold', 'noisy', 'oneshot']:
        split_scores = []
        for sample in eval_data['per_split'][split]['per_sample']:
            if sample.get('success') and sample.get('prediction'):
                score = sample['prediction'].get('verifier_score')
                if score is not None:
                    split_scores.append(score)
        scores[split] = split_scores
    return scores


# ============================================================
# SECTION 4: FIGURE FUNCTIONS
# ============================================================

# ------------------------------
# STAGE 1: CONTRASTIVE LEARNING
# ------------------------------

def fig4_encoder_performance(output_dir: Path):
    """
    Fig 4: Encoder Retrieval Performance
    Grouped bar chart: Baseline vs Contrastive-Trained for Trigger and Action
    FIXED: Legend position to avoid overlapping with value labels
    """
    fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    metrics = ENCODER_GOLD['metrics']
    x = np.arange(len(metrics))
    width = 0.35

    # Left: Trigger Encoder
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, ENCODER_GOLD['trigger']['baseline'], width,
                    label='Pretrained', color=COLORS['baseline'], edgecolor='black', linewidth=0.5,
                    hatch=HATCHES['baseline'])
    bars2 = ax1.bar(x + width/2, ENCODER_GOLD['trigger']['trained'], width,
                    label='Contrastive-Trained', color=COLORS['trained'], edgecolor='black', linewidth=0.5,
                    hatch=HATCHES['trained'])

    ax1.set_xlabel('Metric')
    ax1.set_ylabel('Score')
    ax1.set_title('(a) Trigger Encoder')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.set_ylim(0, 1.15)  # Extended to make room for labels
    ax1.legend(loc='lower right', frameon=True)  # FIXED: moved to lower right
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))

    # Add value labels on trained bars (above bars, no overlap)
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Right: Action Encoder
    ax2 = axes[1]
    bars3 = ax2.bar(x - width/2, ENCODER_GOLD['action']['baseline'], width,
                    label='Pretrained', color=COLORS['baseline'], edgecolor='black', linewidth=0.5,
                    hatch=HATCHES['baseline'])
    bars4 = ax2.bar(x + width/2, ENCODER_GOLD['action']['trained'], width,
                    label='Contrastive-Trained', color=COLORS['trained'], edgecolor='black', linewidth=0.5,
                    hatch=HATCHES['trained'])

    ax2.set_xlabel('Metric')
    ax2.set_ylabel('Score')
    ax2.set_title('(b) Action Encoder')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.set_ylim(0, 1.15)  # Extended to make room for labels
    ax2.legend(loc='lower right', frameon=True)  # FIXED: moved to lower right
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))

    # Add value labels on trained bars
    for bar in bars4:
        height = bar.get_height()
        ax2.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()

    # Save
    output_path = output_dir / 'fig4_encoder_performance.pdf'
    plt.savefig(output_path)
    plt.savefig(output_path.with_suffix('.png'))
    plt.close()
    print(f"  Saved: {output_path}")


def fig5_training_curves(output_dir: Path):
    """
    Fig 5: Contrastive Training Convergence
    Loss curves for both trigger and action encoders
    Colorful but B&W compatible (different line styles and markers)
    """
    fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    # Colors for train/eval - distinguishable in both color and B&W
    train_color = '#2E86AB'  # Blue
    eval_color = '#A23B72'   # Magenta/Pink

    # Trigger Encoder
    ax1 = axes[0]

    # Filter out None values for plotting
    trigger_train_epochs = [e for e, l in zip(TRAINING_LOSS['trigger']['epochs'],
                                               TRAINING_LOSS['trigger']['train_loss']) if l is not None]
    trigger_train_loss = [l for l in TRAINING_LOSS['trigger']['train_loss'] if l is not None]
    trigger_eval_epochs = [e for e, l in zip(TRAINING_LOSS['trigger']['epochs'],
                                              TRAINING_LOSS['trigger']['eval_loss']) if l is not None]
    trigger_eval_loss = [l for l in TRAINING_LOSS['trigger']['eval_loss'] if l is not None]

    # Colorful with distinct markers for B&W: solid circles for train, dashed squares for eval
    ax1.plot(trigger_train_epochs, trigger_train_loss, 'o-', color=train_color,
             label='Train Loss', markersize=7, markerfacecolor=train_color,
             markeredgecolor='black', markeredgewidth=0.5, linewidth=2)
    ax1.plot(trigger_eval_epochs, trigger_eval_loss, 's--', color=eval_color,
             label='Eval Loss', markersize=7, markerfacecolor=eval_color,
             markeredgecolor='black', markeredgewidth=0.5, linewidth=2)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Contrastive Loss')
    ax1.set_title('(a) Trigger Encoder')
    ax1.legend(loc='upper right', frameon=True, fontsize=9)
    ax1.set_xlim(0, 3.2)
    ax1.set_ylim(0, 1.4)

    # Add annotation for improvement
    ax1.annotate('85.6% reduction', xy=(2.4, 0.14), xytext=(1.5, 0.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2),
                fontsize=9, fontweight='bold', color='black',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=0.9))

    # Action Encoder
    ax2 = axes[1]

    action_train_epochs = [e for e, l in zip(TRAINING_LOSS['action']['epochs'],
                                              TRAINING_LOSS['action']['train_loss']) if l is not None]
    action_train_loss = [l for l in TRAINING_LOSS['action']['train_loss'] if l is not None]
    action_eval_epochs = [e for e, l in zip(TRAINING_LOSS['action']['epochs'],
                                             TRAINING_LOSS['action']['eval_loss']) if l is not None]
    action_eval_loss = [l for l in TRAINING_LOSS['action']['eval_loss'] if l is not None]

    # Colorful with distinct markers for B&W: solid circles for train, dashed squares for eval
    ax2.plot(action_train_epochs, action_train_loss, 'o-', color=train_color,
             label='Train Loss', markersize=7, markerfacecolor=train_color,
             markeredgecolor='black', markeredgewidth=0.5, linewidth=2)
    ax2.plot(action_eval_epochs, action_eval_loss, 's--', color=eval_color,
             label='Eval Loss', markersize=7, markerfacecolor=eval_color,
             markeredgecolor='black', markeredgewidth=0.5, linewidth=2)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Contrastive Loss')
    ax2.set_title('(b) Action Encoder')
    ax2.legend(loc='upper right', frameon=True, fontsize=9)
    ax2.set_xlim(0, 3.2)
    ax2.set_ylim(0, 1.4)

    # Add annotation for improvement
    ax2.annotate('83.6% reduction', xy=(2.53, 0.20), xytext=(1.5, 0.6),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2),
                fontsize=9, fontweight='bold', color='black',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=0.9))

    plt.tight_layout()

    output_path = output_dir / 'fig5_training_curves.pdf'
    plt.savefig(output_path)
    plt.savefig(output_path.with_suffix('.png'))
    plt.close()
    print(f"  Saved: {output_path}")


def fig6_retrieval_quality(output_dir: Path, agentic_metrics: dict):
    """
    Fig 6: Retrieval Quality (Stage 1)
    Context Recall and Precision for Trigger and Action across test sets
    """
    fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    test_sets = ['Gold', 'Noisy', 'One-shot']
    x = np.arange(len(test_sets))
    width = 0.35

    # Left: Context Recall
    ax1 = axes[0]
    trigger_recall = [agentic_metrics['gold']['context_recall_trigger'],
                      agentic_metrics['noisy']['context_recall_trigger'],
                      agentic_metrics['oneshot']['context_recall_trigger']]
    action_recall = [agentic_metrics['gold']['context_recall_action'],
                     agentic_metrics['noisy']['context_recall_action'],
                     agentic_metrics['oneshot']['context_recall_action']]

    bars1 = ax1.bar(x - width/2, trigger_recall, width,
                    label='Trigger', color=COLORS['trigger'], edgecolor='black', linewidth=0.5,
                    hatch=HATCHES['trigger'])
    bars2 = ax1.bar(x + width/2, action_recall, width,
                    label='Action', color=COLORS['action'], edgecolor='black', linewidth=0.5,
                    hatch=HATCHES['action'])

    ax1.set_xlabel('Test Set')
    ax1.set_ylabel('Context Recall@5')
    ax1.set_title('(a) Context Recall')
    ax1.set_xticks(x)
    ax1.set_xticklabels(test_sets)
    ax1.set_ylim(0, 1.1)
    ax1.legend(loc='lower right', frameon=True)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.0%}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    # Right: Context Precision
    ax2 = axes[1]
    trigger_precision = [agentic_metrics['gold']['context_precision_trigger'],
                         agentic_metrics['noisy']['context_precision_trigger'],
                         agentic_metrics['oneshot']['context_precision_trigger']]
    action_precision = [agentic_metrics['gold']['context_precision_action'],
                        agentic_metrics['noisy']['context_precision_action'],
                        agentic_metrics['oneshot']['context_precision_action']]

    bars3 = ax2.bar(x - width/2, trigger_precision, width,
                    label='Trigger', color=COLORS['trigger'], edgecolor='black', linewidth=0.5,
                    hatch=HATCHES['trigger'])
    bars4 = ax2.bar(x + width/2, action_precision, width,
                    label='Action', color=COLORS['action'], edgecolor='black', linewidth=0.5,
                    hatch=HATCHES['action'])

    ax2.set_xlabel('Test Set')
    ax2.set_ylabel('Context Precision')
    ax2.set_title('(b) Context Precision')
    ax2.set_xticks(x)
    ax2.set_xticklabels(test_sets)
    ax2.set_ylim(0, 1.1)
    ax2.legend(loc='lower right', frameon=True)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    # Add value labels
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.0%}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    output_path = output_dir / 'fig6_retrieval_quality.pdf'
    plt.savefig(output_path)
    plt.savefig(output_path.with_suffix('.png'))
    plt.close()
    print(f"  Saved: {output_path}")


# ------------------------------
# STAGE 2: MULTI-AGENT SELECTION
# ------------------------------

def fig7_selection_performance(output_dir: Path, agentic_metrics: dict):
    """
    Fig 7: Multi-Agent Selection Results (Stage 2)
    Performance across Gold, Noisy, One-shot test sets
    """
    fig, ax = plt.subplots(figsize=FIG_DOUBLE_TALL)

    metrics = ['Goal\nAccuracy', 'Trigger\nAccuracy', 'Action\nAccuracy', 'Joint\nAccuracy']
    x = np.arange(len(metrics))
    width = 0.25

    gold_values = [agentic_metrics['gold']['goal_acc'],
                   agentic_metrics['gold']['trigger_acc'],
                   agentic_metrics['gold']['action_acc'],
                   agentic_metrics['gold']['joint_acc']]
    noisy_values = [agentic_metrics['noisy']['goal_acc'],
                    agentic_metrics['noisy']['trigger_acc'],
                    agentic_metrics['noisy']['action_acc'],
                    agentic_metrics['noisy']['joint_acc']]
    oneshot_values = [agentic_metrics['oneshot']['goal_acc'],
                      agentic_metrics['oneshot']['trigger_acc'],
                      agentic_metrics['oneshot']['action_acc'],
                      agentic_metrics['oneshot']['joint_acc']]

    bars1 = ax.bar(x - width, gold_values, width, label='Gold',
                   color=COLORS['gold'], edgecolor='black', linewidth=0.5,
                   hatch=HATCHES['gold'])
    bars2 = ax.bar(x, noisy_values, width, label='Noisy',
                   color=COLORS['noisy'], edgecolor='black', linewidth=0.5,
                   hatch=HATCHES['noisy'])
    bars3 = ax.bar(x + width, oneshot_values, width, label='One-shot',
                   color=COLORS['oneshot'], edgecolor='black', linewidth=0.5,
                   hatch=HATCHES['oneshot'])

    ax.set_xlabel('Metric')
    ax.set_ylabel('Accuracy')
    ax.set_title('Multi-Agent Selection Performance Across Test Sets')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='lower right', frameon=True)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.0%}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=7)

    plt.tight_layout()

    output_path = output_dir / 'fig7_selection_performance.pdf'
    plt.savefig(output_path)
    plt.savefig(output_path.with_suffix('.png'))
    plt.close()
    print(f"  Saved: {output_path}")


def fig8_quality_metrics(output_dir: Path, agentic_metrics: dict):
    """
    Fig 8: Quality Metrics (Stage 2)
    Faithfulness and Topic Adherence across test sets
    """
    fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    test_sets = ['Gold', 'Noisy', 'One-shot']
    x = np.arange(len(test_sets))
    width = 0.5

    # Left: Faithfulness
    ax1 = axes[0]
    faithfulness = [agentic_metrics['gold']['faithfulness'],
                    agentic_metrics['noisy']['faithfulness'],
                    agentic_metrics['oneshot']['faithfulness']]

    colors_faith = [COLORS['gold'], COLORS['noisy'], COLORS['oneshot']]
    hatches_faith = [HATCHES['gold'], HATCHES['noisy'], HATCHES['oneshot']]
    bars1 = ax1.bar(x, faithfulness, width, color=colors_faith, edgecolor='black', linewidth=0.5)
    # Apply hatches to individual bars (matplotlib doesn't support hatch list directly)
    for bar, hatch in zip(bars1, hatches_faith):
        bar.set_hatch(hatch)

    ax1.set_xlabel('Test Set')
    ax1.set_ylabel('Faithfulness Score')
    ax1.set_title('(a) Faithfulness')
    ax1.set_xticks(x)
    ax1.set_xticklabels(test_sets)
    ax1.set_ylim(0, 0.7)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.1%}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Right: Topic Adherence
    ax2 = axes[1]
    adherence = [agentic_metrics['gold']['topic_adherence'],
                 agentic_metrics['noisy']['topic_adherence'],
                 agentic_metrics['oneshot']['topic_adherence']]

    bars2 = ax2.bar(x, adherence, width, color=colors_faith, edgecolor='black', linewidth=0.5)
    # Apply hatches to individual bars
    for bar, hatch in zip(bars2, hatches_faith):
        bar.set_hatch(hatch)

    ax2.set_xlabel('Test Set')
    ax2.set_ylabel('Topic Adherence Score')
    ax2.set_title('(b) Topic Adherence')
    ax2.set_xticks(x)
    ax2.set_xticklabels(test_sets)
    ax2.set_ylim(0, 1.0)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{height:.1%}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()

    output_path = output_dir / 'fig8_quality_metrics.pdf'
    plt.savefig(output_path)
    plt.savefig(output_path.with_suffix('.png'))
    plt.close()
    print(f"  Saved: {output_path}")


# ------------------------------
# ABLATION STUDIES
# ------------------------------

def fig9_layer_freezing(output_dir: Path):
    """
    Fig 9: Layer Freezing Ablation - Two clean panels using R@5 (consistent with Fig 11)
    (a) R@5 accuracy - layer freeze slightly better
    (b) Trainable parameters - 82% reduction with layer freezing
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

    methods = ['Full\nContrastive', 'Layer\nFreeze']
    x = np.arange(len(methods))
    width = 0.32

    # Data from LORA_ABLATION (R@5, k=5 evaluation)
    trigger_r5 = LAYER_FREEZING['trigger_r5']  # [0.92, 0.94]
    action_r5 = LAYER_FREEZING['action_r5']    # [0.90, 0.91]
    trainable_params = LAYER_FREEZING['trainable_pct']  # [100.0, 18.0]

    # Panel (a): R@5 Accuracy
    bars1 = ax1.bar(x - width/2, trigger_r5, width,
                    label='Trigger', color=COLORS['trigger'], edgecolor='black', linewidth=0.5,
                    hatch=HATCHES['trigger'])
    bars2 = ax1.bar(x + width/2, action_r5, width,
                    label='Action', color=COLORS['action'], edgecolor='black', linewidth=0.5,
                    hatch=HATCHES['action'])

    ax1.set_ylabel('R@5 Accuracy', fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=9)
    ax1.set_ylim(0.8, 1.0)
    ax1.legend(loc='lower right', frameon=True, fontsize=9)
    ax1.set_title('(a) Retrieval Accuracy (R@5)', fontsize=10, fontweight='bold')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.0%}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 2), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Add improvement annotation
    ax1.annotate('+2-4pp\nbetter', xy=(1, 0.95), xytext=(1.3, 0.88),
                fontsize=9, ha='center', va='center', fontweight='bold', color='#2E7D32',
                arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.5))

    # Panel (b): Trainable Parameters
    colors_param = [COLORS['baseline'], COLORS['trained']]
    bars3 = ax2.bar(x, trainable_params, width=0.5, color=colors_param,
                    edgecolor='black', linewidth=0.5)
    bars3[1].set_hatch(HATCHES['trained'])

    ax2.set_ylabel('Trainable Parameters (%)', fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=9)
    ax2.set_ylim(0, 115)
    ax2.set_title('(b) Parameter Efficiency', fontsize=10, fontweight='bold')

    # Add value labels
    for bar, val in zip(bars3, trainable_params):
        ax2.annotate(f'{val:.0f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 2), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add reduction arrow
    ax2.annotate('', xy=(1, 25), xytext=(0, 95),
                arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=2.5,
                               connectionstyle='arc3,rad=-0.2'))
    ax2.text(0.5, 55, '82%\nfewer', fontsize=10, ha='center', va='center',
             fontweight='bold', color='#D32F2F')

    plt.tight_layout()

    output_path = output_dir / 'fig9_layer_freezing.pdf'
    plt.savefig(output_path)
    plt.savefig(output_path.with_suffix('.png'))
    plt.close()
    print(f"  Saved: {output_path}")


def fig10_component_ablation(output_dir: Path):
    """
    Fig 10: Component Contribution - Incremental Performance Gains
    Shows how each component improves the system:
    - Baseline: Pretrained model
    - + Contrastive: Domain adaptation improves retrieval
    - + Layer Freezing: Efficient training (same R@1, fewer params)
    - + Multi-Agent: Final selection boosts joint accuracy
    """
    fig, ax = plt.subplots(figsize=(8.5, 4.5))

    components = COMPONENT_ABLATION['components']
    x = np.arange(len(components))
    width = 0.25  # Width for 3 bars

    # 3 metrics: Trigger, Action, Joint (removed Semantic)
    bars1 = ax.bar(x - width, COMPONENT_ABLATION['trigger_r1'], width,
                   label='Trigger Acc', color=COLORS['trigger'], edgecolor='black', linewidth=0.5,
                   hatch=HATCHES['trigger'])
    bars2 = ax.bar(x, COMPONENT_ABLATION['action_r1'], width,
                   label='Action Acc', color=COLORS['action'], edgecolor='black', linewidth=0.5,
                   hatch=HATCHES['action'])
    bars3 = ax.bar(x + width, COMPONENT_ABLATION['joint_accuracy'], width,
                   label='Joint Acc', color=COLORS['trained'], edgecolor='black', linewidth=0.5,
                   hatch=HATCHES['trained'])

    ax.set_xlabel('Configuration')
    ax.set_ylabel('Accuracy')
    ax.set_title('Component Contribution to End-to-End Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(components, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper left', frameon=True, ncol=3, fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    # Add value labels for Joint Accuracy (key metric)
    for i, bar in enumerate(bars3):
        height = bar.get_height()
        ax.annotate(f'{height:.0%}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9, fontweight='bold', color='black')

    # Add annotation showing final improvement
    ax.annotate('+74pp Joint Acc\n(7%→81%)',
                xy=(3, 0.81), xytext=(3.5, 0.55),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2),
                fontsize=9, color='black', ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.9))

    plt.tight_layout()

    output_path = output_dir / 'fig10_component_ablation.pdf'
    plt.savefig(output_path)
    plt.savefig(output_path.with_suffix('.png'))
    plt.close()
    print(f"  Saved: {output_path}")


def fig11_lora_ablation(output_dir: Path):
    """
    Fig 11: LoRA vs Layer Freezing Ablation
    Shows: Layer Freezing outperforms LoRA for TAP domain adaptation
    Key insight: LoRA's low-rank approximation doesn't capture domain-specific patterns
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    methods = LORA_ABLATION['methods']
    x = np.arange(len(methods))

    # Left: Joint R@5 Accuracy (main metric)
    ax1 = axes[0]
    colors_joint = [COLORS['baseline'], COLORS['secondary'], COLORS['secondary'],
                    COLORS['secondary'], COLORS['trained']]
    bars1 = ax1.bar(x, LORA_ABLATION['joint_r5'], width=0.6,
                    color=colors_joint, edgecolor='black', linewidth=0.5)

    # Highlight the winner
    bars1[4].set_hatch(HATCHES['trained'])
    bars1[0].set_hatch(HATCHES['baseline'])
    for i in [1, 2, 3]:
        bars1[i].set_hatch(HATCHES['secondary'])

    ax1.set_xlabel('Adaptation Method')
    ax1.set_ylabel('Joint R@5 Accuracy')
    ax1.set_title('(a) Retrieval Accuracy by Method')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=8)
    ax1.set_ylim(0, 1.0)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        fontw = 'bold' if height >= 0.85 else 'normal'
        ax1.annotate(f'{height:.0%}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight=fontw)

    # Right: Efficiency vs Accuracy trade-off
    ax2 = axes[1]

    # Scatter plot: x = trainable params (%), y = joint accuracy
    scatter_colors = [COLORS['baseline'], COLORS['secondary'], COLORS['secondary'],
                      COLORS['secondary'], COLORS['trained']]
    markers = ['s', 'o', 'o', 'o', '^']  # square, circles, triangle

    for i, (pct, acc, method) in enumerate(zip(LORA_ABLATION['trainable_pct'],
                                                LORA_ABLATION['joint_r5'],
                                                methods)):
        ax2.scatter(pct, acc, c=scatter_colors[i], s=150, marker=markers[i],
                   edgecolors='black', linewidths=1, zorder=5)
        # Label each point
        offset = (5, 5) if i != 0 else (-40, 5)
        ax2.annotate(method.replace('\n', ' '), (pct, acc),
                    textcoords="offset points", xytext=offset,
                    fontsize=8, ha='left')

    ax2.set_xlabel('Trainable Parameters (%)')
    ax2.set_ylabel('Joint R@5 Accuracy')
    ax2.set_title('(b) Efficiency vs Accuracy Trade-off')
    ax2.set_xlim(-5, 110)
    ax2.set_ylim(0.60, 0.95)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax2.axhline(y=0.82, color='gray', linestyle='--', alpha=0.5, label='Full Fine-tune baseline')
    ax2.legend(loc='lower right', fontsize=8)

    # Add annotation for Layer Freeze
    ax2.annotate('Best: Layer Freeze\n(18% params, 86% acc)',
                xy=(18, 0.86), xytext=(40, 0.75),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2),
                fontsize=9, color='black', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.9))

    plt.tight_layout()

    output_path = output_dir / 'fig11_lora_ablation.pdf'
    plt.savefig(output_path)
    plt.savefig(output_path.with_suffix('.png'))
    plt.close()
    print(f"  Saved: {output_path}")


# ============================================================
# SECTION 5: MAIN - GENERATE ALL FIGURES
# ============================================================

def main():
    """Generate all figures for FARM paper."""
    print("=" * 60)
    print("FARM Paper Figure Generation")
    print("Q1 Journal Style")
    print("=" * 60)

    # Setup
    setup_journal_style()

    # Output directory
    script_dir = Path(__file__).parent.parent.parent
    output_dir = script_dir / 'latex' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory: {output_dir}")
    print("-" * 60)

    # Load agentic data
    print("\nLoading eval.json...")
    eval_data = load_eval_json()
    agentic_metrics = extract_agentic_metrics(eval_data)
    verifier_scores = extract_verifier_scores(eval_data)

    # Generate figures
    print("\nGenerating figures...")

    # STAGE 1: CONTRASTIVE LEARNING
    print("\n" + "=" * 40)
    print("STAGE 1: CONTRASTIVE LEARNING")
    print("=" * 40)

    print("\n[1/7] Fig 4: Encoder Retrieval Performance")
    fig4_encoder_performance(output_dir)

    print("\n[2/7] Fig 5: Contrastive Training Convergence")
    fig5_training_curves(output_dir)

    print("\n[3/7] Fig 6: Retrieval Quality (Context Recall & Precision)")
    fig6_retrieval_quality(output_dir, agentic_metrics)

    # STAGE 2: MULTI-AGENT SELECTION
    print("\n" + "=" * 40)
    print("STAGE 2: MULTI-AGENT SELECTION")
    print("=" * 40)

    print("\n[4/7] Fig 7: Selection Performance")
    fig7_selection_performance(output_dir, agentic_metrics)

    print("\n[5/7] Fig 8: Quality Metrics (Faithfulness & Topic Adherence)")
    fig8_quality_metrics(output_dir, agentic_metrics)

    # ABLATION STUDIES
    print("\n" + "=" * 40)
    print("ABLATION STUDIES")
    print("=" * 40)

    print("\n[6/8] Fig 9: Layer Freezing Ablation")
    fig9_layer_freezing(output_dir)

    print("\n[7/8] Fig 10: Component Contribution")
    fig10_component_ablation(output_dir)

    print("\n[8/8] Fig 11: LoRA vs Layer Freezing Ablation")
    fig11_lora_ablation(output_dir)

    print("\n" + "=" * 60)
    print("All figures generated successfully!")
    print("=" * 60)

    # Summary
    print("\nGenerated files:")
    for fig_file in sorted(output_dir.glob('fig*.pdf')):
        print(f"  - {fig_file.name}")


if __name__ == '__main__':
    main()
