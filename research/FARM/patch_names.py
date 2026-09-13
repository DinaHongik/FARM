# Result files were named without the base model, so a BGE run silently
# overwrote the MiniLM run at the same loss/seed/epochs/data/k.
p = "rerank_v2.py"
s = open(p).read()
old = 'open(f"logs/v2_{A.loss}_s{A.seed}_e{A.epochs}_{A.data}_k{A.topk}.json", "w"))'
new = 'open(f"logs/v2_{A.loss}_s{A.seed}_e{A.epochs}_{A.data}_k{A.topk}_{A.base.split(chr(47))[-1]}.json", "w"))'
assert old in s, "filename anchor not found"
s = s.replace(old, new)
old2 = 'json.dump({"loss": A.loss,'
new2 = 'json.dump({"base": A.base, "topk": A.topk, "loss": A.loss,'
assert old2 in s, "payload anchor not found"
s = s.replace(old2, new2)
old3 = 'output_dir=f"runs/rerank_v2/{A.loss}_s{A.seed}_e{A.epochs}"'
new3 = 'output_dir=f"runs/rerank_v2/{A.loss}_s{A.seed}_e{A.epochs}_k{A.topk}_{A.base.split(chr(47))[-1]}"'
assert old3 in s, "output_dir anchor not found"
s = s.replace(old3, new3)
open(p, "w").write(s)
print("patched all three")
