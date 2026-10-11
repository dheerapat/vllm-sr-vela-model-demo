"""Load Vela 2.0 4B on a small-RAM machine.

The official loader casts every parameter to FP32 (~17 GB) and builds the model
on the CPU. This machine has 6.7 GB RAM and a 16 GB GPU. This loader streams each
safetensors shard to the GPU and keeps the stored dtypes: backbone bf16, heads
FP32. That is the bf16 parity setting of the release (see PARITY.md:
ref_fp32_r == ref_bf16).
"""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "Vela-2.0-4B"


def load_model(model_dir=MODEL_DIR, device="cuda"):
    model_dir = Path(model_dir).resolve()
    sys.path.insert(0, str(model_dir))  # standalone modeling_vela2 import
    import modeling_vela2 as M  # ty: ignore[unresolved-import]
    from safetensors.torch import load_file

    def load_weights_to_device(model, d):
        d = Path(d)
        idx = d / "model.safetensors.index.json"
        files = sorted(set(json.loads(idx.read_text())["weight_map"].values()))
        params = dict(model.named_parameters())
        seen = set()
        for f in files:
            shard = load_file(str(d / f), device=device)
            for k, v in shard.items():
                if k not in params:
                    raise KeyError(f"unexpected tensor {k} in {f}")
                if tuple(params[k].shape) != tuple(v.shape):
                    raise ValueError(f"shape mismatch {k}: {tuple(v.shape)}")
                with torch.no_grad():
                    params[k].data = v  # drop the empty CPU storage, keep GPU tensor
                seen.add(k)
            del shard
            torch.cuda.empty_cache()
        missing = sorted(set(params) - seen)
        if missing:
            raise KeyError(f"missing tensors: {missing[:5]} (+{len(missing) - 5})")
        return model

    M.load_weights = load_weights_to_device
    model = M.Vela2DecoderModel.from_pretrained(str(model_dir))
    return model.to(device).eval()  # moves the (small) rotary buffers to GPU
