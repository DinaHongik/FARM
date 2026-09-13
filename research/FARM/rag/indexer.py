"""
Qdrant Indexer for FARM RAG System

Builds collections for both BASELINE and FINE-TUNED models:
- farm_triggers_baseline / farm_triggers_finetuned
- farm_actions_baseline / farm_actions_finetuned

Usage:
  python -m rag.indexer --baseline    # Index with pretrained model
  python -m rag.indexer --finetuned   # Index with fine-tuned models
  python -m rag.indexer --all         # Index both (default)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# Handle imports for both direct run and module import
try:
    from .config import FARMConfig, config, get_baseline_config, get_finetuned_config
    from .embeddings import EmbeddingModel
except ImportError:
    from config import FARMConfig, config, get_baseline_config, get_finetuned_config
    from embeddings import EmbeddingModel


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
    channel = filter_code.split('.')[0]
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
            elif field_name != "status":
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
    if channel:
        text_parts.append(f"[{channel}]")
    if category:
        text_parts.append(f"[{category}]")
    text_parts.append(f"{service_name}.")
    text_parts.append(description)
    if field_labels:
        text_parts.append(f"Trigger fields: {', '.join(field_labels)}.")
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

    # Extract field labels
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
    if channel:
        text_parts.append(f"[{channel}]")
    if category:
        text_parts.append(f"[{category}]")
    text_parts.append(f"{service_name}.")
    text_parts.append(description)
    if field_labels:
        text_parts.append(f"Action fields: {', '.join(field_labels)}.")
    if field_parts:
        text_parts.append(f"Requires: {', '.join(field_parts)}.")

    return " ".join(text_parts)


def extract_channel(item: Dict[str, Any], is_trigger: bool = True) -> str:
    """Extract channel from item for metadata."""
    api_info = item.get("api_info", {})

    if is_trigger:
        ingredients = api_info.get("Ingredients", {})
        for name, info in ingredients.items():
            if isinstance(info, dict) and "Filter code" in info:
                return extract_channel_from_filter_code(info["Filter code"])
    else:
        action_fields = api_info.get("Action fields", {})
        for name, info in action_fields.items():
            if isinstance(info, dict) and "Filter code method" in info:
                return extract_channel_from_filter_code(info["Filter code method"])

    return ""


def load_json(filepath: Path) -> List[Dict[str, Any]]:
    """Load JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


class QdrantIndexer:
    """Indexer for building Qdrant collections."""

    def __init__(self, farm_config: FARMConfig = None):
        """
        Initialize the indexer.

        Args:
            farm_config: Configuration. Uses default if not provided.
        """
        self.config = farm_config or config

        # Use local storage (no Docker needed)
        storage_path = self.config.qdrant.storage_path
        if storage_path:
            # Persistent local storage
            self.client = QdrantClient(path=storage_path)
        else:
            # In-memory (for testing)
            self.client = QdrantClient(":memory:")

        # Create embedding model with this config (baseline or fine-tuned)
        self.embedding_model = EmbeddingModel(self.config.embedding)
        self.is_finetuned = self.config.embedding.use_finetuned

    def create_collection(self, collection_name: str, recreate: bool = True):
        """
        Create a Qdrant collection.

        Args:
            collection_name: Name of the collection.
            recreate: If True, delete existing collection first.
        """
        if recreate:
            try:
                self.client.delete_collection(collection_name)
                print(f"Deleted existing collection: {collection_name}")
            except Exception:
                pass

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=self.config.embedding.dimension,
                distance=Distance.COSINE
            )
        )
        print(f"Created collection: {collection_name} (dim={self.config.embedding.dimension})")

    def index_triggers(self, recreate: bool = True):
        """
        Index all triggers into Qdrant.

        Args:
            recreate: If True, recreate the collection.
        """
        collection_name = self.config.qdrant.triggers_collection

        # Load data
        print(f"\nLoading triggers from: {self.config.data.triggers_path}")
        triggers = load_json(self.config.data.triggers_path)
        print(f"Loaded {len(triggers)} triggers")

        # Create collection
        self.create_collection(collection_name, recreate=recreate)

        # Prepare texts for embedding
        print("Preparing embedding texts...")
        texts = [extract_trigger_text(t) for t in triggers]

        # Generate embeddings using appropriate encoder
        if self.is_finetuned:
            print("Generating embeddings with FINE-TUNED trigger encoder...")
            embeddings = self.embedding_model.encode_triggers(texts, show_progress=True)
        else:
            print("Generating embeddings with BASELINE (pretrained) encoder...")
            embeddings = self.embedding_model.encode(texts, show_progress=True)

        # Prepare points for upsert
        print("Preparing points for upsert...")
        points = []
        for i, (trigger, embedding) in enumerate(zip(triggers, embeddings)):
            channel = extract_channel(trigger, is_trigger=True)
            payload = {
                "service_name": trigger.get("service_name", ""),
                "category": trigger.get("category", ""),
                "channel": channel,
                "description": trigger.get("description", ""),
                "api_info": trigger.get("api_info", {}),
                "embedding_text": texts[i]
            }
            points.append(PointStruct(
                id=i,
                vector=embedding.tolist(),
                payload=payload
            ))

        # Upsert in batches
        print("Upserting to Qdrant...")
        batch_size = 100
        for i in tqdm(range(0, len(points), batch_size), desc="Upserting"):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=collection_name,
                points=batch
            )

        print(f"Indexed {len(triggers)} triggers into '{collection_name}'")

    def index_actions(self, recreate: bool = True):
        """
        Index all actions into Qdrant.

        Args:
            recreate: If True, recreate the collection.
        """
        collection_name = self.config.qdrant.actions_collection

        # Load data
        print(f"\nLoading actions from: {self.config.data.actions_path}")
        actions = load_json(self.config.data.actions_path)
        print(f"Loaded {len(actions)} actions")

        # Create collection
        self.create_collection(collection_name, recreate=recreate)

        # Prepare texts for embedding
        print("Preparing embedding texts...")
        texts = [extract_action_text(a) for a in actions]

        # Generate embeddings using appropriate encoder
        if self.is_finetuned:
            print("Generating embeddings with FINE-TUNED action encoder...")
            embeddings = self.embedding_model.encode_actions(texts, show_progress=True)
        else:
            print("Generating embeddings with BASELINE (pretrained) encoder...")
            embeddings = self.embedding_model.encode(texts, show_progress=True)

        # Prepare points for upsert
        print("Preparing points for upsert...")
        points = []
        for i, (action, embedding) in enumerate(zip(actions, embeddings)):
            channel = extract_channel(action, is_trigger=False)
            payload = {
                "service_name": action.get("service_name", ""),
                "category": action.get("category", ""),
                "channel": channel,
                "description": action.get("description", ""),
                "api_info": action.get("api_info", {}),
                "embedding_text": texts[i]
            }
            points.append(PointStruct(
                id=i,
                vector=embedding.tolist(),
                payload=payload
            ))

        # Upsert in batches
        print("Upserting to Qdrant...")
        batch_size = 100
        for i in tqdm(range(0, len(points), batch_size), desc="Upserting"):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=collection_name,
                points=batch
            )

        print(f"Indexed {len(actions)} actions into '{collection_name}'")

    def index_all(self, recreate: bool = True):
        """
        Index both triggers and actions.

        Args:
            recreate: If True, recreate collections.
        """
        mode = "FINE-TUNED" if self.is_finetuned else "BASELINE (pretrained)"
        print("=" * 60)
        print(f"FARM RAG Indexer - {mode}")
        print("=" * 60)
        print(f"Mode: {mode}")
        print(f"Qdrant storage: {self.config.qdrant.storage_path}")
        print(f"Triggers collection: {self.config.qdrant.triggers_collection}")
        print(f"Actions collection: {self.config.qdrant.actions_collection}")
        print(f"Base model: {self.config.embedding.model_name}")
        print(f"Dimension: {self.config.embedding.dimension}")

        self.index_triggers(recreate=recreate)
        self.index_actions(recreate=recreate)

        print("\n" + "=" * 60)
        print("Indexing complete!")
        print("=" * 60)

    def get_collection_info(self):
        """Print info about existing collections."""
        print("\nCollection Info:")
        for name in [self.config.qdrant.triggers_collection, self.config.qdrant.actions_collection]:
            try:
                info = self.client.get_collection(name)
                print(f"  {name}: {info.points_count} points")
            except Exception as e:
                print(f"  {name}: Not found")


def main():
    """Main function to run indexing with CLI support."""
    parser = argparse.ArgumentParser(
        description="FARM RAG Indexer - Build Qdrant collections for retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m rag.indexer --baseline     # Index with pretrained model only
  python -m rag.indexer --finetuned    # Index with fine-tuned models only
  python -m rag.indexer --all          # Index both (default)
  python -m rag.indexer                # Same as --all
        """
    )
    parser.add_argument(
        "--baseline", action="store_true",
        help="Index using BASELINE (pretrained) model"
    )
    parser.add_argument(
        "--finetuned", action="store_true",
        help="Index using FINE-TUNED models"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Index both baseline and fine-tuned (default)"
    )

    args = parser.parse_args()

    # Default to --all if no option specified
    if not args.baseline and not args.finetuned and not args.all:
        args.all = True

    # Run baseline indexing
    if args.baseline or args.all:
        print("\n" + "=" * 70)
        print("INDEXING BASELINE (PRETRAINED)")
        print("=" * 70)
        baseline_config = get_baseline_config()
        indexer = QdrantIndexer(baseline_config)
        indexer.index_all(recreate=True)
        indexer.get_collection_info()
        # Close client to release lock before opening another
        indexer.client.close()
        del indexer

    # Run fine-tuned indexing
    if args.finetuned or args.all:
        print("\n" + "=" * 70)
        print("INDEXING FINE-TUNED")
        print("=" * 70)
        finetuned_config = get_finetuned_config()
        indexer = QdrantIndexer(finetuned_config)
        indexer.index_all(recreate=True)
        indexer.get_collection_info()
        indexer.client.close()
        del indexer

    print("\n" + "=" * 70)
    print("ALL INDEXING COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
