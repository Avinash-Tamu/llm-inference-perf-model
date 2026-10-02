"""Smoke test: GPT-3 175B, one decoder layer, decode step, single A100 (GA100 config).
Run from repo root:  python myexp/smoke_test.py
Expected (verified): bs=1 ~2.1 ms simulated, bs=64 ~5.3 ms simulated (ctx=2048)."""
import sys, time
sys.path.insert(0, ".")
from design_space_exploration.dse import template_to_system, read_architecture_template
from software_model.transformer import TransformerBlockAutoRegressionTP
from software_model.utils import data_type_dict, Tensor

specs = read_architecture_template("configs/GA100.json")
specs["device_count"] = 1
system = template_to_system(specs)
d, h, ctx = 12288, 96, 2048
for bs in [1, 64]:
    m = TransformerBlockAutoRegressionTP(d_model=d, n_heads=h, device_count=1,
                                         data_type=data_type_dict["fp16"])
    _ = m(Tensor([bs, 1, d], data_type_dict["fp16"]), ctx)
    t = time.time()
    rl = m.roofline_model(system)
    sim = m.compile_and_simulate(system, "heuristic-GPU")
    print(f"DECODE bs={bs} ctx={ctx}: roofline={rl*1e3:.3f} ms  "
          f"simulated={sim*1e3:.3f} ms  (wall {time.time()-t:.1f}s)")
