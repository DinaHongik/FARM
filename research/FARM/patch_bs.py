# BGE at k=20 needs 320 pairs/step at bs=16 and OOMs a 32GB V100.
# Expose bs/accum so the EFFECTIVE batch stays 16 lists, keeping every run comparable.
p = "rerank_v2.py"
s = open(p).read()
old = 'ap.add_argument("--maxlen", type=int, default=320)'
new = ('ap.add_argument("--maxlen", type=int, default=320)\n'
       'ap.add_argument("--bs", type=int, default=16, help="per-device batch in LISTS")\n'
       'ap.add_argument("--accum", type=int, default=1,\n'
       '                help="grad accumulation; bs*accum is the effective batch, keep it at 16")')
assert old in s and '--bs' not in s, "arg anchor bad"
s = s.replace(old, new)
old2 = 'num_train_epochs=A.epochs, per_device_train_batch_size=16,'
new2 = ('num_train_epochs=A.epochs, per_device_train_batch_size=A.bs,\n'
        '    gradient_accumulation_steps=A.accum,')
assert old2 in s, "batch anchor bad"
s = s.replace(old2, new2)
open(p, "w").write(s)
print("patched")
