# Test Sets

This directory should contain the test sets for evaluation. These files are **available upon request** from the authors.

## Required Files

| File | Description | Samples |
|------|-------------|---------|
| `gold.json` | Clear, well-formed descriptions (>=50 characters) | 100 |
| `noisy.json` | Vague, ambiguous descriptions (<40 characters) | 100 |
| `oneshot.json` | Applets using rare APIs (<20 occurrences in training) | 100 |

## File Format

Each test file is a JSON array of applet objects:

```json
[
  {
    "description": "When I leave work, remind me to buy groceries",
    "trigger": {
      "service_name": "You exit an area",
      "category": "Location",
      "api_info": {...}
    },
    "action": {
      "service_name": "Send a notification",
      "category": "Notifications",
      "api_info": {...}
    }
  }
]
```

## Creating Your Own Test Sets

Use `data/split_data.py` to create test sets from your own applet data:

```bash
python data/split_data.py --input your_applets.json --output data/
```

The script will automatically categorize applets into gold/noisy/oneshot based on:
- **Gold**: Description length >= 50 characters
- **Noisy**: Description length < 40 characters
- **Oneshot**: APIs appearing < 20 times in training data
