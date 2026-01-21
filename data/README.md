# Data Directory

This directory should contain the dataset files required to run FARM. The datasets are **available upon request** from the authors.

## Required Files

### Training Data (for encoder fine-tuning)

| File | Description | Format |
|------|-------------|--------|
| `iftttt_dataset_full_trigger_action.json` | Full IFTTT applet dataset | JSON array of applet objects |
| `train_applets.json` | Training split (90%) | JSON array of applet objects |

### RAG Corpus (for retrieval)

| File | Description | Format |
|------|-------------|--------|
| `triggers_rag.json` | Trigger API corpus (1,724 APIs) | JSON array |
| `actions_rag.json` | Action API corpus (1,287 APIs) | JSON array |

### Test Sets (for evaluation)

Place these in `data/test/`:

| File | Description | Samples |
|------|-------------|---------|
| `gold.json` | Clear descriptions (>=50 chars) | 100 |
| `noisy.json` | Vague descriptions (<40 chars) | 100 |
| `oneshot.json` | Rare APIs (<20 occurrences) | 100 |

## Data Formats

### Applet Format (Training Data)

```json
{
  "description": "User description of the applet",
  "trigger": {
    "service_name": "Darkness is detected",
    "category": "Smart home & IoT",
    "api_info": {
      "Trigger fields": {},
      "Ingredients": {
        "ingredient_name": {
          "Slug": "...",
          "Filter code": "Channel.Service.Field",
          "Type": "String",
          "Example": "..."
        }
      }
    }
  },
  "action": {
    "service_name": "Add row to spreadsheet",
    "category": "Productivity",
    "api_info": {
      "Action fields": {
        "field_name": {
          "Type": "Text field",
          "Required": true,
          "Helper text": "..."
        }
      }
    }
  }
}
```

### RAG Corpus Format (triggers_rag.json / actions_rag.json)

```json
[
  {
    "service_name": "Darkness is detected",
    "category": "Smart home & IoT",
    "description": "This trigger fires when...",
    "api_info": {
      "Trigger fields": {},
      "Ingredients": {...}
    }
  }
]
```

### Test Set Format

```json
[
  {
    "description": "User query for the applet",
    "trigger": {...},
    "action": {...}
  }
]
```

## Requesting the Dataset

To request access to the dataset, please contact the authors:

**Email:** young.yoon@hongik.ac.kr

Please include:
- Your name and affiliation
- Intended use case
- Agreement to use for research purposes only

## Building Your Own Dataset

If you prefer to use your own data:

1. **Collect applets** from IFTTT or similar platforms
2. **Format** according to the schemas above
3. **Split** into train/test using `split_data.py`
4. **Generate** test sets using your own criteria

The code is designed to work with any dataset following the above format.
