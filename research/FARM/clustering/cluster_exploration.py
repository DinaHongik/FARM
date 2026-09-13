"""
Cluster Exploration for FARM Project
=====================================
This script explores the structure of trigger and action APIs
to determine the best clustering approach (HDBSCAN vs K-Means).

Steps:
1. Load triggers and actions datasets
2. Create semantic embedding text for each API
3. Generate embeddings using sentence-transformers
4. Visualize with UMAP to understand data structure
5. Compare HDBSCAN vs K-Means clustering
6. Save visualizations and analysis results
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Clustering and dimensionality reduction
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
import hdbscan
import umap

# Sentence embeddings
from sentence_transformers import SentenceTransformer

# Set style for visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    """Configuration for the clustering exploration."""
    # Paths
    project_root: Path = Path(__file__).parent.parent
    # Data is located at /workspace/work/data/data/ (outside FARM folder)
    data_dir: Path = project_root.parent / "data" / "data"
    viz_dir: Path = Path(__file__).parent / "visualizations"

    # Data files
    triggers_file: str = "triggers_rag.json"
    actions_file: str = "actions_rag.json"

    # Embedding model
    embedding_model: str = "BAAI/bge-base-en-v1.5"

    # HDBSCAN parameter sweep
    hdbscan_min_cluster_sizes: List[int] = None
    hdbscan_min_samples: List[int] = None

    # K-Means parameter sweep
    kmeans_k_range: Tuple[int, int] = (10, 60)

    # UMAP parameters
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_metric: str = "cosine"

    def __post_init__(self):
        if self.hdbscan_min_cluster_sizes is None:
            self.hdbscan_min_cluster_sizes = [10, 15, 20, 30, 50]
        if self.hdbscan_min_samples is None:
            self.hdbscan_min_samples = [5, 10, 15]

        # Ensure visualization directory exists
        self.viz_dir.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Data Loading
# =============================================================================

def load_dataset(filepath: Path) -> List[Dict[str, Any]]:
    """Load a JSON dataset from file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_channel_from_filter_code(filter_code: str) -> str:
    """
    Extract channel name from Filter code.

    Examples:
    - "AqaraHomeEU.darknessDetected.Date" -> "AqaraHomeEU"
    - "GoogleSheets.appendToGoogleSpreadsheet.setFilename" -> "GoogleSheets"
    - "_5MinuteCrafts.newVideo.EntryTitle" -> "5MinuteCrafts"
    """
    if not filter_code:
        return ""
    # Split by '.' and take the first part
    channel = filter_code.split('.')[0]
    # Remove leading underscore if present (e.g., "_5MinuteCrafts" -> "5MinuteCrafts")
    if channel.startswith('_'):
        channel = channel[1:]
    return channel


def extract_trigger_text(trigger: Dict[str, Any]) -> str:
    """
    Create rich semantic embedding text for a trigger.

    Format:
    "[{channel}] [{category}] {service_name}. {description}
    Trigger fields: {field_labels}
    Provides: {ingredient_name} ({type}), ..."
    """
    service_name = trigger.get("service_name", "")
    description = trigger.get("description", "")
    category = trigger.get("category", "")
    api_info = trigger.get("api_info", {})

    # Extract channel name from first ingredient's Filter code
    channel = ""
    ingredients = api_info.get("Ingredients", {})
    if ingredients:
        for name, info in ingredients.items():
            if isinstance(info, dict) and "Filter code" in info:
                channel = extract_channel_from_filter_code(info["Filter code"])
                break

    # Extract trigger field labels
    trigger_fields = api_info.get("Trigger fields", {})
    field_labels = []
    if isinstance(trigger_fields, dict):
        for field_name, field_info in trigger_fields.items():
            if isinstance(field_info, dict):
                label = field_info.get("Label", field_name)
                field_labels.append(label)
            elif field_name != "status":  # Skip "No fields for this trigger"
                field_labels.append(field_name)

    # Extract ingredients with types
    ingredient_parts = []
    if ingredients:
        for name, info in ingredients.items():
            if isinstance(info, dict):
                ing_type = info.get("Type", "Unknown")
                ingredient_parts.append(f"{name} ({ing_type})")

    # Build the rich embedding text
    text_parts = []

    # Add channel and category as prefixes for strong semantic signal
    if channel:
        text_parts.append(f"[{channel}]")
    if category:
        text_parts.append(f"[{category}]")

    # Add service name and description
    text_parts.append(f"{service_name}.")
    text_parts.append(description)

    # Add trigger field labels
    if field_labels:
        text_parts.append(f"Trigger fields: {', '.join(field_labels)}.")

    # Add ingredients
    if ingredient_parts:
        text_parts.append(f"Provides: {', '.join(ingredient_parts)}.")

    return " ".join(text_parts)


def extract_action_text(action: Dict[str, Any]) -> str:
    """
    Create rich semantic embedding text for an action.

    Format:
    "[{channel}] [{category}] {service_name}. {description}
    Action fields: {field_labels}
    Requires: {field_name} (required={T/F}), ..."
    """
    service_name = action.get("service_name", "")
    description = action.get("description", "")
    category = action.get("category", "")
    api_info = action.get("api_info", {})

    # Extract channel name from first field's Filter code method
    channel = ""
    action_fields = api_info.get("Action fields", {})
    if action_fields:
        for name, info in action_fields.items():
            if isinstance(info, dict) and "Filter code method" in info:
                channel = extract_channel_from_filter_code(info["Filter code method"])
                break

    # Extract field labels and helper texts
    field_labels = []
    field_parts = []
    if action_fields:
        for name, info in action_fields.items():
            if isinstance(info, dict):
                label = info.get("Label", name)
                field_labels.append(label)
                required = info.get("Required", "false")
                field_parts.append(f"{label} (required={required})")

    # Build the rich embedding text
    text_parts = []

    # Add channel and category as prefixes for strong semantic signal
    if channel:
        text_parts.append(f"[{channel}]")
    if category:
        text_parts.append(f"[{category}]")

    # Add service name and description
    text_parts.append(f"{service_name}.")
    text_parts.append(description)

    # Add action field labels
    if field_labels:
        text_parts.append(f"Action fields: {', '.join(field_labels)}.")

    # Add required fields info
    if field_parts:
        text_parts.append(f"Requires: {', '.join(field_parts)}.")

    return " ".join(text_parts)


# =============================================================================
# Embedding Generation
# =============================================================================

def create_embeddings(texts: List[str], model_name: str) -> np.ndarray:
    """Generate embeddings for a list of texts using sentence-transformers."""
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"Generating embeddings for {len(texts)} texts...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # For cosine similarity
    )

    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings


# =============================================================================
# UMAP Visualization
# =============================================================================

def compute_umap(embeddings: np.ndarray, config: Config) -> np.ndarray:
    """Reduce embeddings to 2D using UMAP."""
    print("Computing UMAP projection...")
    reducer = umap.UMAP(
        n_neighbors=config.umap_n_neighbors,
        min_dist=config.umap_min_dist,
        metric=config.umap_metric,
        random_state=42
    )
    umap_embeddings = reducer.fit_transform(embeddings)
    print(f"UMAP shape: {umap_embeddings.shape}")
    return umap_embeddings


def plot_umap_by_category(
    umap_embeddings: np.ndarray,
    categories: List[str],
    title: str,
    save_path: Path
):
    """Plot UMAP colored by original IFTTT categories."""
    fig, ax = plt.subplots(figsize=(14, 10))

    unique_categories = list(set(categories))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_categories)))
    color_map = {cat: colors[i] for i, cat in enumerate(unique_categories)}

    for cat in unique_categories:
        mask = [c == cat for c in categories]
        points = umap_embeddings[mask]
        ax.scatter(
            points[:, 0], points[:, 1],
            c=[color_map[cat]],
            label=f"{cat} ({sum(mask)})",
            alpha=0.6,
            s=30
        )

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_umap_density(
    umap_embeddings: np.ndarray,
    title: str,
    save_path: Path
):
    """Plot UMAP with density estimation to visualize cluster structure."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Scatter plot
    axes[0].scatter(
        umap_embeddings[:, 0],
        umap_embeddings[:, 1],
        alpha=0.5,
        s=20,
        c='steelblue'
    )
    axes[0].set_title(f"{title} - Scatter", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("UMAP Dimension 1")
    axes[0].set_ylabel("UMAP Dimension 2")

    # Density plot (KDE)
    sns.kdeplot(
        x=umap_embeddings[:, 0],
        y=umap_embeddings[:, 1],
        ax=axes[1],
        cmap="viridis",
        fill=True,
        levels=20
    )
    axes[1].set_title(f"{title} - Density", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("UMAP Dimension 1")
    axes[1].set_ylabel("UMAP Dimension 2")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================================
# Clustering Analysis
# =============================================================================

def analyze_hdbscan(
    embeddings: np.ndarray,
    config: Config,
    dataset_name: str
) -> pd.DataFrame:
    """
    Run HDBSCAN with different parameters and analyze results.
    Returns a DataFrame with clustering statistics for each configuration.
    """
    results = []

    for min_cluster_size in config.hdbscan_min_cluster_sizes:
        for min_samples in config.hdbscan_min_samples:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='euclidean',
                cluster_selection_method='eom'
            )
            labels = clusterer.fit_predict(embeddings)

            # Calculate statistics
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_outliers = sum(labels == -1)
            outlier_pct = 100 * n_outliers / len(labels)

            # Cluster size distribution (excluding outliers)
            cluster_sizes = [sum(labels == i) for i in range(n_clusters)]

            if n_clusters > 0:
                size_min = min(cluster_sizes)
                size_max = max(cluster_sizes)
                size_mean = np.mean(cluster_sizes)
                size_std = np.std(cluster_sizes)
            else:
                size_min = size_max = size_mean = size_std = 0

            # Silhouette score (only if we have valid clusters and non-outliers)
            non_outlier_mask = labels != -1
            if n_clusters > 1 and sum(non_outlier_mask) > n_clusters:
                sil_score = silhouette_score(
                    embeddings[non_outlier_mask],
                    labels[non_outlier_mask]
                )
            else:
                sil_score = -1

            results.append({
                'dataset': dataset_name,
                'method': 'HDBSCAN',
                'min_cluster_size': min_cluster_size,
                'min_samples': min_samples,
                'n_clusters': n_clusters,
                'n_outliers': n_outliers,
                'outlier_pct': round(outlier_pct, 2),
                'size_min': size_min,
                'size_max': size_max,
                'size_mean': round(size_mean, 2),
                'size_std': round(size_std, 2),
                'silhouette': round(sil_score, 4)
            })

    return pd.DataFrame(results)


def analyze_kmeans(
    embeddings: np.ndarray,
    config: Config,
    dataset_name: str
) -> pd.DataFrame:
    """
    Run K-Means with different K values and analyze results.
    Returns a DataFrame with clustering statistics for each K.
    """
    results = []
    k_min, k_max = config.kmeans_k_range

    for k in range(k_min, k_max + 1, 5):  # Step by 5
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        # Cluster size distribution
        cluster_sizes = [sum(labels == i) for i in range(k)]

        size_min = min(cluster_sizes)
        size_max = max(cluster_sizes)
        size_mean = np.mean(cluster_sizes)
        size_std = np.std(cluster_sizes)

        # Silhouette score
        sil_score = silhouette_score(embeddings, labels)

        # Calinski-Harabasz score
        ch_score = calinski_harabasz_score(embeddings, labels)

        # Inertia (within-cluster sum of squares)
        inertia = kmeans.inertia_

        results.append({
            'dataset': dataset_name,
            'method': 'K-Means',
            'k': k,
            'n_clusters': k,
            'n_outliers': 0,
            'outlier_pct': 0,
            'size_min': size_min,
            'size_max': size_max,
            'size_mean': round(size_mean, 2),
            'size_std': round(size_std, 2),
            'silhouette': round(sil_score, 4),
            'calinski_harabasz': round(ch_score, 2),
            'inertia': round(inertia, 2)
        })

    return pd.DataFrame(results)


def plot_clustering_comparison(
    hdbscan_results: pd.DataFrame,
    kmeans_results: pd.DataFrame,
    dataset_name: str,
    save_path: Path
):
    """Create comparison plots for HDBSCAN vs K-Means."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Number of clusters
    ax1 = axes[0, 0]
    ax1.bar(
        range(len(hdbscan_results)),
        hdbscan_results['n_clusters'],
        alpha=0.7,
        label='HDBSCAN'
    )
    ax1.axhline(
        y=kmeans_results['k'].median(),
        color='red',
        linestyle='--',
        label=f'K-Means median K={int(kmeans_results["k"].median())}'
    )
    ax1.set_xlabel("HDBSCAN Configuration")
    ax1.set_ylabel("Number of Clusters")
    ax1.set_title("Number of Clusters by Configuration")
    ax1.legend()

    # Plot 2: Outlier percentage for HDBSCAN
    ax2 = axes[0, 1]
    ax2.bar(
        range(len(hdbscan_results)),
        hdbscan_results['outlier_pct'],
        alpha=0.7,
        color='orange'
    )
    ax2.set_xlabel("HDBSCAN Configuration")
    ax2.set_ylabel("Outlier Percentage (%)")
    ax2.set_title("HDBSCAN Outlier Percentage")
    ax2.axhline(y=10, color='red', linestyle='--', label='10% threshold')
    ax2.legend()

    # Plot 3: Silhouette score comparison
    ax3 = axes[1, 0]
    ax3.plot(
        kmeans_results['k'],
        kmeans_results['silhouette'],
        'b-o',
        label='K-Means',
        markersize=6
    )
    # Add HDBSCAN best silhouette as horizontal line
    best_hdbscan_sil = hdbscan_results[hdbscan_results['silhouette'] > 0]['silhouette'].max()
    if best_hdbscan_sil > 0:
        ax3.axhline(
            y=best_hdbscan_sil,
            color='green',
            linestyle='--',
            label=f'Best HDBSCAN: {best_hdbscan_sil:.4f}'
        )
    ax3.set_xlabel("K (Number of Clusters)")
    ax3.set_ylabel("Silhouette Score")
    ax3.set_title("Silhouette Score vs K")
    ax3.legend()

    # Plot 4: Cluster size distribution (K-Means)
    ax4 = axes[1, 1]
    ax4.fill_between(
        kmeans_results['k'],
        kmeans_results['size_min'],
        kmeans_results['size_max'],
        alpha=0.3,
        label='Min-Max Range'
    )
    ax4.plot(
        kmeans_results['k'],
        kmeans_results['size_mean'],
        'g-o',
        label='Mean Size',
        markersize=6
    )
    ax4.set_xlabel("K (Number of Clusters)")
    ax4.set_ylabel("Cluster Size")
    ax4.set_title("K-Means Cluster Size Distribution")
    ax4.legend()

    fig.suptitle(f"Clustering Analysis: {dataset_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_elbow_curve(
    kmeans_results: pd.DataFrame,
    dataset_name: str,
    save_path: Path
):
    """Plot elbow curve for K-Means."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        kmeans_results['k'],
        kmeans_results['inertia'],
        'b-o',
        markersize=8
    )
    ax.set_xlabel("K (Number of Clusters)", fontsize=12)
    ax.set_ylabel("Inertia (Within-cluster sum of squares)", fontsize=12)
    ax.set_title(f"Elbow Curve - {dataset_name}", fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_cluster_visualization(
    umap_embeddings: np.ndarray,
    labels: np.ndarray,
    title: str,
    save_path: Path,
    show_outliers: bool = True
):
    """Plot UMAP with cluster labels."""
    fig, ax = plt.subplots(figsize=(12, 9))

    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

    # Create color map
    colors = plt.cm.tab20(np.linspace(0, 1, max(20, n_clusters)))

    for label in sorted(unique_labels):
        mask = labels == label
        if label == -1:
            if show_outliers:
                ax.scatter(
                    umap_embeddings[mask, 0],
                    umap_embeddings[mask, 1],
                    c='gray',
                    alpha=0.3,
                    s=20,
                    label=f'Outliers ({sum(mask)})'
                )
        else:
            ax.scatter(
                umap_embeddings[mask, 0],
                umap_embeddings[mask, 1],
                c=[colors[label % len(colors)]],
                alpha=0.6,
                s=30,
                label=f'Cluster {label} ({sum(mask)})'
            )

    ax.set_title(f"{title}\n({n_clusters} clusters)", fontsize=14, fontweight='bold')
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")

    # Only show legend if not too many clusters
    if n_clusters <= 20:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================================
# Main Execution
# =============================================================================

def process_dataset(
    data: List[Dict],
    extract_func,
    dataset_name: str,
    config: Config
) -> Dict[str, Any]:
    """Process a single dataset (triggers or actions)."""
    print(f"\n{'='*60}")
    print(f"Processing: {dataset_name}")
    print(f"{'='*60}")

    # Extract categories for visualization
    categories = [item.get("category", "Unknown") for item in data]
    print(f"Total items: {len(data)}")
    print(f"Unique categories: {len(set(categories))}")

    # Show category distribution
    cat_counts = Counter(categories)
    print("\nCategory distribution:")
    for cat, count in cat_counts.most_common(10):
        print(f"  {cat}: {count}")

    # Create embedding texts
    print("\nCreating embedding texts...")
    texts = [extract_func(item) for item in data]

    # Show sample texts
    print("\nSample embedding texts:")
    for i in range(min(3, len(texts))):
        print(f"  [{i}] {texts[i][:150]}...")

    # Generate embeddings
    embeddings = create_embeddings(texts, config.embedding_model)

    # Compute UMAP
    umap_embeddings = compute_umap(embeddings, config)

    # Save UMAP visualizations
    plot_umap_by_category(
        umap_embeddings,
        categories,
        f"{dataset_name} - Original IFTTT Categories",
        config.viz_dir / f"{dataset_name.lower()}_umap_categories.png"
    )

    plot_umap_density(
        umap_embeddings,
        f"{dataset_name}",
        config.viz_dir / f"{dataset_name.lower()}_umap_density.png"
    )

    # Analyze HDBSCAN
    print("\nAnalyzing HDBSCAN configurations...")
    hdbscan_results = analyze_hdbscan(embeddings, config, dataset_name)

    # Analyze K-Means
    print("Analyzing K-Means configurations...")
    kmeans_results = analyze_kmeans(embeddings, config, dataset_name)

    # Plot comparisons
    plot_clustering_comparison(
        hdbscan_results,
        kmeans_results,
        dataset_name,
        config.viz_dir / f"{dataset_name.lower()}_clustering_comparison.png"
    )

    plot_elbow_curve(
        kmeans_results,
        dataset_name,
        config.viz_dir / f"{dataset_name.lower()}_elbow_curve.png"
    )

    # Find best configurations
    print(f"\n--- {dataset_name} Results ---")

    # Best HDBSCAN (lowest outlier % with reasonable cluster count)
    valid_hdbscan = hdbscan_results[
        (hdbscan_results['n_clusters'] >= 10) &
        (hdbscan_results['outlier_pct'] < 30)
    ]
    if not valid_hdbscan.empty:
        best_hdbscan = valid_hdbscan.loc[valid_hdbscan['silhouette'].idxmax()]
        print(f"\nBest HDBSCAN:")
        print(f"  min_cluster_size={best_hdbscan['min_cluster_size']}, min_samples={best_hdbscan['min_samples']}")
        print(f"  Clusters: {best_hdbscan['n_clusters']}, Outliers: {best_hdbscan['outlier_pct']}%")
        print(f"  Silhouette: {best_hdbscan['silhouette']}")

        # Visualize best HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=int(best_hdbscan['min_cluster_size']),
            min_samples=int(best_hdbscan['min_samples']),
            metric='euclidean'
        )
        hdbscan_labels = clusterer.fit_predict(embeddings)
        plot_cluster_visualization(
            umap_embeddings,
            hdbscan_labels,
            f"{dataset_name} - Best HDBSCAN",
            config.viz_dir / f"{dataset_name.lower()}_best_hdbscan.png"
        )
    else:
        print("\nNo valid HDBSCAN configuration found (too many outliers)")
        best_hdbscan = None

    # Best K-Means (highest silhouette)
    best_kmeans = kmeans_results.loc[kmeans_results['silhouette'].idxmax()]
    print(f"\nBest K-Means:")
    print(f"  K={best_kmeans['k']}")
    print(f"  Silhouette: {best_kmeans['silhouette']}")
    print(f"  Cluster sizes: min={best_kmeans['size_min']}, max={best_kmeans['size_max']}, mean={best_kmeans['size_mean']}")

    # Visualize best K-Means
    kmeans = KMeans(n_clusters=int(best_kmeans['k']), random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(embeddings)
    plot_cluster_visualization(
        umap_embeddings,
        kmeans_labels,
        f"{dataset_name} - Best K-Means (K={int(best_kmeans['k'])})",
        config.viz_dir / f"{dataset_name.lower()}_best_kmeans.png"
    )

    return {
        'dataset_name': dataset_name,
        'n_items': len(data),
        'n_categories': len(set(categories)),
        'embeddings': embeddings,
        'umap_embeddings': umap_embeddings,
        'categories': categories,
        'hdbscan_results': hdbscan_results,
        'kmeans_results': kmeans_results,
        'best_hdbscan': best_hdbscan,
        'best_kmeans': best_kmeans
    }


def generate_summary_report(
    triggers_results: Dict,
    actions_results: Dict,
    config: Config
):
    """Generate a summary report of the clustering exploration."""
    report_path = config.viz_dir / "clustering_exploration_report.txt"

    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("CLUSTERING EXPLORATION REPORT\n")
        f.write("FARM Project - Trigger-Action API Clustering\n")
        f.write("=" * 70 + "\n\n")

        for results in [triggers_results, actions_results]:
            f.write(f"\n{'='*50}\n")
            f.write(f"{results['dataset_name']}\n")
            f.write(f"{'='*50}\n")
            f.write(f"Total items: {results['n_items']}\n")
            f.write(f"Original IFTTT categories: {results['n_categories']}\n\n")

            f.write("HDBSCAN Analysis:\n")
            f.write("-" * 40 + "\n")
            f.write(results['hdbscan_results'].to_string(index=False))
            f.write("\n\n")

            f.write("K-Means Analysis:\n")
            f.write("-" * 40 + "\n")
            f.write(results['kmeans_results'].to_string(index=False))
            f.write("\n\n")

            if results['best_hdbscan'] is not None:
                f.write("Best HDBSCAN Configuration:\n")
                f.write(f"  min_cluster_size: {results['best_hdbscan']['min_cluster_size']}\n")
                f.write(f"  min_samples: {results['best_hdbscan']['min_samples']}\n")
                f.write(f"  Clusters: {results['best_hdbscan']['n_clusters']}\n")
                f.write(f"  Outliers: {results['best_hdbscan']['outlier_pct']}%\n")
                f.write(f"  Silhouette: {results['best_hdbscan']['silhouette']}\n\n")

            f.write("Best K-Means Configuration:\n")
            f.write(f"  K: {results['best_kmeans']['k']}\n")
            f.write(f"  Silhouette: {results['best_kmeans']['silhouette']}\n")
            f.write(f"  Size range: {results['best_kmeans']['size_min']} - {results['best_kmeans']['size_max']}\n\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("RECOMMENDATION\n")
        f.write("=" * 70 + "\n")
        f.write("Based on the analysis above, review the visualizations to determine:\n")
        f.write("1. Does the data show clear density-based clusters? -> HDBSCAN\n")
        f.write("2. Is the data more uniformly distributed? -> K-Means\n")
        f.write("3. Check outlier percentage for HDBSCAN - if >20%, consider K-Means\n")
        f.write("4. Compare silhouette scores between methods\n")
        f.write("\nVisualization files saved in: {}\n".format(config.viz_dir))

    print(f"\nReport saved: {report_path}")


def main():
    """Main execution function."""
    print("=" * 70)
    print("FARM Project - Cluster Exploration")
    print("=" * 70)

    # Initialize configuration
    config = Config()
    print(f"\nConfiguration:")
    print(f"  Data directory: {config.data_dir}")
    print(f"  Visualization directory: {config.viz_dir}")
    print(f"  Embedding model: {config.embedding_model}")

    # Load datasets
    print("\nLoading datasets...")
    triggers_path = config.data_dir / config.triggers_file
    actions_path = config.data_dir / config.actions_file

    triggers_data = load_dataset(triggers_path)
    actions_data = load_dataset(actions_path)

    print(f"Loaded {len(triggers_data)} triggers")
    print(f"Loaded {len(actions_data)} actions")

    # Process triggers
    triggers_results = process_dataset(
        triggers_data,
        extract_trigger_text,
        "Triggers",
        config
    )

    # Process actions
    actions_results = process_dataset(
        actions_data,
        extract_action_text,
        "Actions",
        config
    )

    # Generate summary report
    generate_summary_report(triggers_results, actions_results, config)

    print("\n" + "=" * 70)
    print("Exploration complete!")
    print(f"Check visualizations in: {config.viz_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
