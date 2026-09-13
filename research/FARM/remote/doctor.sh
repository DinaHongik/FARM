#!/bin/bash
# Environment health check, run on the server.
# Lives here rather than inline in farm.ps1 because PowerShell does not treat
# backslash as an escape character - embedded quotes in a remote command string
# get re-parsed locally and break.

WD=/raid/session/aicontents/farm
cd "$WD" || exit 1

echo "-- host --"
hostname
uptime | tr -s ' '

echo
echo "-- gpus --"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

echo
echo "-- python env --"
if [ -x .venv/bin/python ]; then
  .venv/bin/python - <<'PY'
import importlib
for mod, label in (("torch","torch"), ("transformers","transformers"),
                   ("sentence_transformers","sentence-transformers"),
                   ("qdrant_client","qdrant-client"), ("langgraph","langgraph")):
    try:
        m = importlib.import_module(mod)
        print(f"  {label:<22} {getattr(m,'__version__','?')}")
    except Exception as e:
        print(f"  {label:<22} MISSING ({type(e).__name__})")
try:
    import torch
    print(f"  cuda available         {torch.cuda.is_available()} ({torch.cuda.device_count()} devices)")
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        print(f"  device capability      sm_{cap[0]}{cap[1]}  (V100=sm_70: use fp16, NOT bf16)")
except Exception as e:
    print("  torch check failed:", e)
PY
else
  echo "  .venv MISSING"
fi

echo
echo "-- ollama --"
V=$(curl -s --max-time 5 http://127.0.0.1:11434/api/version 2>/dev/null)
if [ -n "$V" ]; then
  echo "  serving: $V"
  OLLAMA_MODELS=$WD/.ollama_models "$HOME/.local/ollama/bin/ollama" list 2>/dev/null | tail -n +2 | sed 's/^/  /'
else
  echo "  NOT RUNNING   (start it:  farm.ps1 run ollama)"
fi

echo
echo "-- huggingface token --"
if [ -s ~/.cache/huggingface/token ]; then
  echo "  present"
else
  echo "  MISSING - training fails on the gated google/embeddinggemma-300m repo"
  echo "  fix:  farm.ps1 token hf_xxxxxxxx"
fi

echo
echo "-- data splits --"
for d in data/test data/test_dedup; do
  if [ -d "$d" ]; then
    n=$(for f in "$d"/*.json; do python3 -c "import json,sys;print(len(json.load(open(sys.argv[1],encoding='utf-8'))))" "$f" 2>/dev/null; done | paste -sd/ -)
    echo "  $d  ($n records)"
  else
    echo "  $d  (absent)"
  fi
done

echo
echo "-- disk --"
df -h "$WD" | tail -1 | awk '{print "  "$4" free of "$2}'
du -sh "$WD" --exclude=models --exclude=.ollama_models 2>/dev/null | awk '{print "  workdir: "$1" (excl. models + ollama)"}'
