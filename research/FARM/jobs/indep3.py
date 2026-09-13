"""Audit 3: run main() offline over all 299 with a stubbed call(); subset bias; join defect."""
import json, importlib.util, sys, io, contextlib
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

sys.argv = ["arms_fixed.py", "STUB", "300", "8"]
NEW = load("nf", "jobs/arms_fixed.py")
NEW.MODEL, NEW.N, NEW.WORKERS = "STUB", 300, 8

# Stub the network: return a plausible reply that omits the choice keys for
# half the samples, to exercise the D5 path, and a null/blank binding for some.
import itertools, threading
ctr = itertools.count(); lock = threading.Lock()
def stub_call(prompt, with_choice, timeout=150):
    with lock: i = next(ctr)
    if i % 7 == 0:
        return None, 0.01, "BADJSON"
    body = {"trigger_bindings": [{"field":"x","source":"static","value":None}],
            "action_bindings": [{"field":"y","source":"ingredient","value":"  "}]}
    if with_choice and i % 3 == 0:
        pass                       # choice keys DELIBERATELY absent
    elif with_choice:
        body = {"trigger_choice": 1, "action_choice": 2, **body}
    return body, 0.02, None
NEW.call = stub_call
buf = io.StringIO()
try:
    with contextlib.redirect_stdout(buf):
        NEW.main()
    print("MAIN RAN OK\n")
except Exception as e:
    print("MAIN CRASHED:", type(e).__name__, e)
print(buf.getvalue())
