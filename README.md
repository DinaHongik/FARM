# FARM: Field-Aware Resolution Model for Trigger-Action Programming

## Overview

Trigger-Action Programming (TAP) is a declarative paradigm for event-driven automation, powering platforms like IFTTT, Zapier, Microsoft Power Automate, and Apple Shortcuts. Traditionally, users manually composed IF-THEN rules by selecting triggers and actions. Recently, LLM-driven systems have shifted this toward natural language specifications, where models must infer intent, select services, and produce executable configurations.

However, AI-based TAP builders often fail on ambiguous requests—users phrase the same intent differently, provide incomplete context, or encode implicit preferences. Current systems may generate inconsistent variants, select wrong parameters, or produce applets that are syntactically valid but semantically misaligned.

FARM addresses these challenges using domain-adapted dual encoders for retrieval and a multi-agent system for resolution and applet generation.

**Dataset available upon request:** young.yoon@hongik.ac.kr

---

## Installation

```bash
# Clone repository
git clone https://github.com/DinaHongik/FARM
cd FARM

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install LLM (required for agents)
ollama run granite4:small-h
```

---

## How to Run

### Step 1: Prepare Data

Place dataset files in `data/` directory:

```
data/
├── iftttt_dataset_full_trigger_action.json   # Training applets
├── triggers_rag.json                          # Trigger API corpus
├── actions_rag.json                           # Action API corpus
└── test/
    ├── gold.json
    ├── noisy.json
    └── oneshot.json
```

### Step 2: Train Encoders

```bash
python -m train.train_trigger
python -m train.train_action
```

### Step 3: Build RAG Index

```bash
python -m rag.indexer --all
```

### Step 4: Run FARM

**Single query:**
```bash
python -m agents.run "When darkness is detected, log to spreadsheet" --verbose
```

**Batch evaluation:**
```bash
python -m eval.run_agentic_eval --splits gold --limit 5 --verbose
```

---

## Project Structure

```
FARM/
├── train/           # Encoder training
├── rag/             # RAG retrieval system
├── agents/          # Multi-agent pipeline
├── eval/            # Evaluation scripts
├── data/            # Dataset (not included)
└── results/         # Output directory
```

---

## License

MIT License
