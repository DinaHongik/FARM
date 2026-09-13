# Using a local LLM

Install Ollama using its [official quickstart](https://docs.ollama.com/quickstart), start the server, and pull a model suitable for your machine. The API server normally listens on `http://127.0.0.1:11434`; its [chat endpoint](https://docs.ollama.com/api/chat) accepts the model, messages, and generation options used by FARM.

```bash
# Run in a separate terminal if the Ollama service is not already running.
ollama serve
```

In the FARM terminal:

```bash
read -rp 'Local Ollama model ID: ' FARM_LOCAL_MODEL
export FARM_LOCAL_MODEL
ollama pull "$FARM_LOCAL_MODEL"
ollama list
farm run --provider ollama --model "$FARM_LOCAL_MODEL" --limit 1 --output outputs/local-smoke
farm run --provider ollama --model "$FARM_LOCAL_MODEL" --output outputs/local-100
```

Use the exact locally installed tag shown by `ollama list`. Model memory requirements depend on its architecture, quantization, and context length; select a model that fits your hardware. A local model does not require an API key. The release sends no authentication header to the local server.

The recovered completed workflow disables thinking, sets temperature to zero and seed to 9052026, requests JSON, and uses at most 8,192 generated tokens per call. If your local model rejects a thinking option, add `--think omit`. Use a different output directory when changing this option. JSON requests are parsed and validated; a parseable response is not automatically semantically correct.

For a different local port:

```bash
farm run --provider ollama --host http://127.0.0.1:11435 --model "$FARM_LOCAL_MODEL" --output outputs/local-port-11435
```

If the model is on another computer, tunnel its Ollama port to localhost using your own SSH account:

```bash
ssh -N -L 11435:127.0.0.1:11434 your-user@your-server
```

Then use the port-11435 command above. No DGX hostname, account, or credential is required by the public runner. The original Ollama Cloud transport is preserved under `research/binding/scripts/cloud.py` for historical reproduction; the public `ollama` option is local-only.

Interrupted runs resume from the same output directory. The output includes the exact provider/model choice and a separate `summary.json`; it never overwrites the paper tables. These examples have supplied endpoints, so this run evaluates the configuration path rather than full-catalog retrieval.
