#!/usr/bin/env python3
"""Export the Fold-2 TCNRegModel checkpoint to a fixed-shape FP32 ONNX graph.

Read-only inputs:
  * original PolyU project run.py and data/split_byid/test
  * original Reg_Model/Cla_Model checkpoints
  * Fold-2 TCN checkpoint

Writes only under experiments/tcn_reg_20261120/onnx_fp32_fold2_20261121/.
No classifier, quantization, Vela build, firmware, or flash operations are done.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def import_helper(helper_path: Path):
    spec = importlib.util.spec_from_file_location("tcn_fold2_export_helper", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import helper module: {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TCNExportWrapper(nn.Module):
    """Expose only ONNX tensor inputs; omit the Python-only device argument."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        x_values: torch.Tensor,
        x_mark: torch.Tensor,
        worker_features_18: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(
            x_values,
            x_mark_enc=x_mark,
            one_hot_x=worker_features_18,
        )


def sensor_valid_counts(x: torch.Tensor) -> list[int]:
    """Count non-sentinel values per sensor in a [1,900,6] test input."""
    missing = (x == -9997) | (x == -9998) | (x == -9999)
    return (~missing).sum(dim=1).reshape(-1).to(torch.int64).tolist()


def make_sample_inputs(dataset: Any, index: int):
    item = dataset[index]
    if len(item) != 6:
        raise RuntimeError(f"Expected 6 dataset outputs, got {len(item)}")
    corrupted, _target, _artificial_mask, marks, worker, _labels = item
    x = corrupted.detach().cpu().to(torch.float32).unsqueeze(0).contiguous()
    m = marks.detach().cpu().to(torch.float32).unsqueeze(0).contiguous()
    w = worker[:18].detach().cpu().to(torch.float32).unsqueeze(0).contiguous()
    if tuple(x.shape) != (1, 900, 6):
        raise RuntimeError(f"Unexpected sequence input shape: {tuple(x.shape)}")
    if tuple(m.shape) != (1, 900, 3):
        raise RuntimeError(f"Unexpected time-mark input shape: {tuple(m.shape)}")
    if tuple(w.shape) != (1, 18):
        raise RuntimeError(f"Unexpected worker-feature input shape: {tuple(w.shape)}")
    return x, m, w


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261121)
    ap.add_argument("--group-column", default="worker_id")
    ap.add_argument("--atol", type=float, default=2e-4)
    ap.add_argument("--rtol", type=float, default=2e-4)
    args = ap.parse_args()

    project = args.project.resolve()
    exp = project / "experiments" / "tcn_reg_20261120"
    helper_path = exp / "train_tcn_reg_testdev_experiment.py"
    checkpoint = (
        exp / "leave_one_worker_out_20261121" /
        "fold_02_heldout_2c624232cdd2" / "best_tcn_reg.pth"
    )
    data_dir = project / "data" / "split_byid"
    run_py = project / "run.py"
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = exp / f"onnx_fp32_fold2_{run_tag}"
    onnx_path = out_dir / "heatstroke_tcn_reg_fold2_fp32.onnx"

    print("===== FOLD-2 TCN FP32 ONNX EXPORT =====")
    print("Timestamp:", datetime.now().astimezone().isoformat())
    print("Python:", sys.version.replace("\n", " "))
    print("Platform:", platform.platform())
    print("PyTorch:", torch.__version__)
    print("Project:", project)
    print("Helper:", helper_path)
    print("Checkpoint:", checkpoint)
    print("Output directory:", out_dir)

    for required in (helper_path, checkpoint, data_dir / "test_seq.feather",
                     data_dir / "test_non_seq.csv", run_py):
        if not required.exists():
            raise FileNotFoundError(f"Required input missing: {required}")
    if out_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing export directory: {out_dir}. "
            "Move/rename it manually before a deliberate repeat."
        )

    import onnx
    import onnxruntime as ort
    print("ONNX:", onnx.__version__)
    print("ONNX Runtime:", ort.__version__)

    out_dir.mkdir(parents=True, exist_ok=False)
    helper = import_helper(helper_path)
    project_run = helper.import_project(project)

    # Deterministic real dataset examples for trace and independent checks.
    dataset = helper.ImputationDataset(
        project_run, data_dir, "test", args.seed, args.group_column
    )
    if len(dataset) < 1:
        raise RuntimeError("The split_byid/test dataset is empty")
    dataset.set_epoch(0)
    indices = sorted(set([0, len(dataset)//4, len(dataset)//2,
                          (3*len(dataset))//4, len(dataset)-1]))
    print("Dataset samples:", len(dataset))
    print("Deterministic verification indices:", indices)

    model = helper.TCNRegModel(project_run, channels=32, kernel_size=31).cpu()
    state = helper.load_checkpoint(checkpoint)
    incompat = model.load_state_dict(state, strict=True)
    if incompat.missing_keys or incompat.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch: {incompat}")
    model.eval()
    wrapper = TCNExportWrapper(model).eval()

    print("Checkpoint SHA256:", sha256_file(checkpoint))
    print("Helper SHA256:", sha256_file(helper_path))
    print("State tensors:", len(state))
    print("Parameter count:", sum(p.numel() for p in model.parameters()))
    print("Model architecture: channels=32, kernel=31, dilations=[1,2]*5")

    sample_inputs = {index: make_sample_inputs(dataset, index) for index in indices}
    print("Valid sensor-value counts in verification samples:")
    for index in indices:
        counts = sensor_valid_counts(sample_inputs[index][0])
        print(f"  sample {index}: {counts}")

    # TCNRegModel has a Python-side empty-sensor guard before adaptive pooling.
    # The legacy ONNX tracer specializes Python branches to the trace example,
    # so choose a trace example with at least one valid value in every channel.
    trace_index = next(
        (i for i in indices if all(c > 0 for c in sensor_valid_counts(sample_inputs[i][0]))),
        None,
    )
    if trace_index is None:
        raise RuntimeError(
            "None of the representative samples has at least one observed value "
            "in every sensor channel; safe tracing of the model's empty-sensor "
            "branch requires a different export strategy."
        )
    first_inputs = sample_inputs[trace_index]
    print("ONNX trace sample index:", trace_index)
    with torch.no_grad():
        eager_probe = wrapper(*first_inputs)
    if tuple(eager_probe.shape) != (1, 900, 6):
        raise RuntimeError(f"Unexpected PyTorch output shape: {tuple(eager_probe.shape)}")
    if not torch.isfinite(eager_probe).all():
        raise RuntimeError("PyTorch probe output contains NaN/Inf")

    export_kwargs = dict(
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["x_values", "x_mark", "worker_features_18"],
        output_names=["reconstructed_values"],
        dynamic_axes=None,
        verbose=False,
    )
    # Retain the established TorchScript exporter where supported.
    import inspect
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    print("\nExporting fixed-shape FP32 ONNX...")
    torch.onnx.export(wrapper, first_inputs, str(onnx_path), **export_kwargs)
    if not onnx_path.is_file() or onnx_path.stat().st_size == 0:
        raise RuntimeError("ONNX export did not create a non-empty model")
    print("ONNX file:", onnx_path)
    print("ONNX bytes:", onnx_path.stat().st_size)
    print("ONNX SHA256:", sha256_file(onnx_path))

    graph = onnx.load(str(onnx_path))
    onnx.checker.check_model(graph)
    opset_imports = {x.domain or "ai.onnx": x.version for x in graph.opset_import}
    op_counts = Counter(node.op_type for node in graph.graph.node)
    print("ONNX checker: PASS")
    print("Opset imports:", opset_imports)
    print("Graph nodes:", len(graph.graph.node))
    print("Operator counts:", json.dumps(dict(sorted(op_counts.items())), indent=2))
    print("Has LSTM op:", op_counts.get("LSTM", 0) > 0)
    for value in list(graph.graph.input) + list(graph.graph.output):
        dims = []
        for dim in value.type.tensor_type.shape.dim:
            dims.append(dim.dim_value if dim.dim_value else (dim.dim_param or "?"))
        print(f"Graph tensor {value.name}: dtype={value.type.tensor_type.elem_type}, shape={dims}")

    print("\nCreating ONNX Runtime CPU session...")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    print("ORT inputs:", [(v.name, v.shape, v.type) for v in session.get_inputs()])
    print("ORT outputs:", [(v.name, v.shape, v.type) for v in session.get_outputs()])

    sample_metrics = []
    all_pass = True
    for index in indices:
        inputs = sample_inputs[index]
        feed = {
            "x_values": inputs[0].numpy(),
            "x_mark": inputs[1].numpy(),
            "worker_features_18": inputs[2].numpy(),
        }
        with torch.no_grad():
            torch_out = wrapper(*inputs).cpu().numpy()
        ort_out = session.run(["reconstructed_values"], feed)[0]
        if torch_out.shape != ort_out.shape:
            raise RuntimeError(
                f"Output shape mismatch at sample {index}: "
                f"PyTorch={torch_out.shape}, ONNX={ort_out.shape}"
            )
        delta = np.abs(torch_out - ort_out)
        close = bool(np.allclose(torch_out, ort_out, atol=args.atol, rtol=args.rtol))
        finite = bool(np.isfinite(ort_out).all())
        passed = close and finite
        all_pass = all_pass and passed
        metrics = {
            "dataset_index": int(index),
            "valid_sensor_value_counts": sensor_valid_counts(inputs[0]),
            "output_shape": list(ort_out.shape),
            "max_abs_error": float(delta.max()),
            "mean_abs_error": float(delta.mean()),
            "rmse": float(np.sqrt(np.mean((torch_out - ort_out) ** 2))),
            "torch_output_min": float(torch_out.min()),
            "torch_output_max": float(torch_out.max()),
            "onnx_output_min": float(ort_out.min()),
            "onnx_output_max": float(ort_out.max()),
            "finite": finite,
            "allclose": close,
            "pass": passed,
        }
        sample_metrics.append(metrics)
        print("COMPARE:", json.dumps(metrics, sort_keys=True))

    manifest = {
        "model": "TCNRegModel Fold-2 FP32 regression/imputation stage only",
        "data_source_for_trace_and_checks": "split_byid/test (diagnostic samples only)",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "helper": str(helper_path),
        "helper_sha256": sha256_file(helper_path),
        "onnx_file": str(onnx_path),
        "onnx_sha256": sha256_file(onnx_path),
        "onnx_bytes": onnx_path.stat().st_size,
        "opset_imports": opset_imports,
        "operator_counts": dict(sorted(op_counts.items())),
        "input_specs": {
            "x_values": [1, 900, 6],
            "x_mark": [1, 900, 3],
            "worker_features_18": [1, 18],
        },
        "output_spec": {"reconstructed_values": [1, 900, 6]},
        "tcn": {"channels": 32, "kernel_size": 31, "dilations": [1, 2] * 5,
                "theoretical_receptive_field": 901},
        "verification": {
            "runtime": "ONNX Runtime CPUExecutionProvider",
            "trace_dataset_index": int(trace_index),
            "atol": args.atol,
            "rtol": args.rtol,
            "all_samples_pass": all_pass,
            "samples": sample_metrics,
        },
        "scope_note": (
            "This is not the full RPE/PSI classifier graph. The existing Cla_Model "
            "is intentionally not included. No quantization, Vela compilation, "
            "firmware build/programming, or flash writes were performed."
        ),
    }
    manifest_path = out_dir / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print("Manifest:", manifest_path)

    if not all_pass:
        raise RuntimeError(
            "ONNX export succeeded but one or more PyTorch-vs-ONNX checks failed. "
            "Do not proceed to TFLite/quantization until the discrepancy is resolved."
        )

    print("\n<EXPORT_RESULT> PASS </EXPORT_RESULT>")
    print("FP32 ONNX export and PyTorch/ONNX Runtime numerical checks: PASS")
    print("No classifier, quantization, Vela, firmware, or flash actions performed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("\n<EXPORT_RESULT> FAIL </EXPORT_RESULT>", file=sys.stderr)
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
