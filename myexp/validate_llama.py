"""Sanity checks for software_model/llama.py against hand calculations."""
import sys; sys.path.insert(0, ".")
from software_model.llama import LLAMA3_70B, LLAMA3_70B_MHA, LlamaDecoderLayerTP
from software_model.utils import Tensor

c = LLAMA3_70B
kv = c.kv_bytes_per_token()
print(f"params: {c.param_count()/1e9:.2f} B (expected ~70.6 B)")
print(f"KV cache/token GQA: {kv/1024:.0f} KB (hand calc 2*80*8*128*2 = 320 KB)")
print(f"KV cache/token MHA: {LLAMA3_70B_MHA.kv_bytes_per_token()/1024:.0f} KB (8x)")
L = LlamaDecoderLayerTP(c, 4)
L(Tensor([1, 1, 8192]), "decode", kv_len=4095)
print(f"per-device weights/layer: {L.weight_bytes/1e6:.1f} MB; x80 = {L.weight_bytes*80/1e9:.2f} GB")
print(f"per-device KV/layer (b=1, 4096 tok): {L.kv_cache_bytes/1e3:.0f} KB -> x4 dev x80 layers /4096 = "
      f"{L.kv_cache_bytes*4*80/4096/1024:.0f} KB/token")
for n, s in L.op_stats().items():
    print(f"  {n:10s} flops={s['flops']:.3e} bytes={s['bytes']:.3e} AI={s['flops']/s['bytes']:.2f}")
