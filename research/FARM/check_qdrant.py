"""Quick script to check Qdrant collections and config."""

from rag.retriever import FARMRetriever
from rag.config import config, get_finetuned_config

print("=" * 60)
print("QDRANT & CONFIG DIAGNOSTIC")
print("=" * 60)

print("\n=== Current Default Config ===")
print(f"  use_finetuned: {config.embedding.use_finetuned}")
print(f"  triggers_collection: {config.qdrant.triggers_collection}")
print(f"  actions_collection: {config.qdrant.actions_collection}")
print(f"  storage_path: {config.qdrant.storage_path}")

print("\n=== Fine-tuned Config ===")
ft_config = get_finetuned_config()
print(f"  use_finetuned: {ft_config.embedding.use_finetuned}")
print(f"  triggers_collection: {ft_config.qdrant.triggers_collection}")
print(f"  actions_collection: {ft_config.qdrant.actions_collection}")

print("\n=== Available Qdrant Collections ===")
r = FARMRetriever()
collections = r.client.get_collections()

if not collections.collections:
    print("  NO COLLECTIONS FOUND!")
else:
    for c in collections.collections:
        info = r.client.get_collection(c.name)
        print(f"  - {c.name}: {info.points_count} points")

print("\n=== Which collection is E2E using? ===")
print(f"  Triggers: {r.config.qdrant.triggers_collection}")
print(f"  Actions: {r.config.qdrant.actions_collection}")
print(f"  is_finetuned: {r.is_finetuned}")

# Check if finetuned collections exist
finetuned_triggers = "farm_triggers_finetuned"
finetuned_actions = "farm_actions_finetuned"
collection_names = [c.name for c in collections.collections]

print("\n=== DIAGNOSIS ===")
if r.is_finetuned:
    print("  ✓ Retriever is using FINE-TUNED mode")
else:
    print("  ✗ Retriever is using BASELINE mode")

if finetuned_triggers in collection_names:
    print(f"  ✓ {finetuned_triggers} exists")
else:
    print(f"  ✗ {finetuned_triggers} MISSING - need to run indexer with --finetuned")

if finetuned_actions in collection_names:
    print(f"  ✓ {finetuned_actions} exists")
else:
    print(f"  ✗ {finetuned_actions} MISSING - need to run indexer with --finetuned")

print("\n" + "=" * 60)
