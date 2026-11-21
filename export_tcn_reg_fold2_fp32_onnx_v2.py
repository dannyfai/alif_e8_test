#!/usr/bin/env python3
"""Export Fold-2 TCNRegModel to FP32 ONNX with explicit compacted-sensor input.

Why v2 exists:
  The training model compacts non-sentinel sensor values with boolean indexing
  and adaptive_avg_pool1d(variable_length, 1800). The legacy ONNX exporter cannot
  export that variable-length adaptive pool reliably, and a traced Python
  empty-channel branch is unsafe for channels that may be entirely missing.

  This exporter moves ONLY that variable-length compaction step outside ONNX.
  It adds a fixed-shape input compact_values [1,1800,6]. All learned model
  weights, including the six SVF Linear layers, remain in the ONNX graph.
  The same compact_values are computed for every PyTorch/ORT test case using
  the exact PyTorch adaptive_avg_pool1d reference operation.

The resulting ONNX graph therefore requires an upstream preprocessor to build
compact_values. A later firmware implementation must reproduce the pooling
formula and include its runtime/memory cost in the SRAM assessment.

No source project files, checkpoints, dataset files, firmware, or flash are
modified. Output is written only under experiments/tcn_reg_20261120/.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
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
import torch.nn.functional as F


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def import_helper(helper_path: Path):
    spec = importlib.util.spec_from_file_location("tcn_fold2_export_helper_v2", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import helper module: {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_compact_input(x: torch.Tensor, compact_seq_len: int = 1800) -> torch.Tensor:
    """Reproduce TCNRegModel's per-channel valid-value compaction exactly.

    Input:  x [B,900,6]
    Output: [B,1800,6]

    This routine intentionally runs outside the ONNX graph. It supports a
    channel with zero valid values by returning an all-zero compact channel.
    """
    if x.ndim != 3 or tuple(x.shape[1:]) != (900, 6):
        raise ValueError(f"Expected x [B,900,6], got {tuple(x.shape)}")
    missing = (x == -9997) | (x == -9998) | (x == -9999)
    batches = []
    for b in range(x.shape[0]):
        channels = []
        for n in range(x.shape[2]):
            values = x[b:b + 1, :, n:n + 1][~missing[b:b + 1, :, n:n + 1]]
            if values.numel() > 0:
                pooled = F.adaptive_avg_pool1d(
                    values.reshape(1, 1, -1), compact_seq_len
                ).reshape(compact_seq_len)
            else:
                pooled = torch.zeros(compact_seq_len, dtype=x.dtype, device=x.device)
            channels.append(pooled)
        batches.append(torch.stack(channels, dim=1))
    return torch.stack(batches, dim=0).contiguous()


def sensor_valid_counts(x: torch.Tensor) -> list[list[int]]:
    missing = (x == -9997) | (x == -9998) | (x == -9999)
    return (~missing).sum(dim=1).to(torch.int64).tolist()


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
    compact = make_compact_input(x)
    if tuple(compact.shape) != (1, 1800, 6):
        raise RuntimeError(f"Unexpected compact input shape: {tuple(compact.shape)}")
    return x, m, w, compact


class TCNExportWrapper(nn.Module):
    """Export-safe model path; variable-length compaction is supplied as input."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        x_values: torch.Tensor,
        x_mark: torch.Tensor,
        worker_features_18: torch.Tensor,
        compact_values: torch.Tensor,
    ) -> torch.Tensor:
        model = self.model
        batch = x_values.shape[0]
        missing_mask = (
            (x_values == -9997) | (x_values == -9998) | (x_values == -9999)
        )

        non_seq_enc_out = model.non_seq_enc(worker_features_18)
        time_and_worker = torch.cat(
            (x_mark, non_seq_enc_out.reshape(batch, -1, 1)), dim=2
        )

        # SVF branch: identical learned Linear layers, fixed-shape input.
        svf_out = [layer(compact_values[:, :, i]) for i, layer in enumerate(model.svf)]
        svf_enc = torch.stack(svf_out, dim=2)
        svf_enc = torch.where(missing_mask, svf_enc, torch.zeros_like(svf_enc))

        # CVF branch rewritten without NaN assignment/nanmean. For a channel
        # with no valid samples, its mean is defined as zero, matching the
        # original nanmean -> nan_to_num behaviour.
        valid_mask = ~missing_mask
        valid_values = torch.where(valid_mask, x_values, torch.zeros_like(x_values))
        valid_count = valid_mask.to(x_values.dtype).sum(dim=1, keepdim=True)
        safe_count = torch.clamp(valid_count, min=1.0)
        channel_means = (valid_values.sum(dim=1, keepdim=True) / safe_count)
        imputed = torch.where(
            missing_mask,
            channel_means.expand(-1, x_values.shape[1], -1),
            x_values,
        )
        # Match the original Reg_Model preprocessing for any exceptional
        # NaN/Inf values after mean imputation.
        imputed = torch.nan_to_num(imputed, nan=0.0, posinf=0.0, neginf=0.0)
        cvf_enc = model.cvf(imputed.permute(0, 2, 1)).permute(0, 2, 1)
        cvf_enc = torch.where(missing_mask, cvf_enc, torch.zeros_like(cvf_enc))

        observed_zeroed = torch.where(missing_mask, torch.zeros_like(x_values), x_values)
        embedding = torch.cat(((svf_enc + cvf_enc) / 2.0, observed_zeroed), dim=2)
        embedding = torch.cat((embedding, time_and_worker), dim=2)

        encoded = model.tcn_in(embedding.transpose(1, 2))
        encoded = F.relu(model.tcn_in_bn(encoded), inplace=False)
        for block in model.tcn_blocks:
            encoded = block(encoded)
        return model.tcn_out(encoded).transpose(1, 2)


def compare_arrays(reference: np.ndarray, candidate: np.ndarray, atol: float, rtol: float):
    delta = np.abs(reference - candidate)
    return {
        "max_abs_error": float(delta.max()),
        "mean_abs_error": float(delta.mean()),
        "rmse": float(np.sqrt(np.mean((reference - candidate) ** 2))),
        "allclose": bool(np.allclose(reference, candidate, atol=atol, rtol=rtol)),
        "finite": bool(np.isfinite(candidate).all()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261121)
    ap.add_argument("--group-column", default="worker_id")
    ap.add_argument("--atol", type=float, default=2e-4)
    ap.add_argument("--rtol", type=float, default=2e-4)
    ap.add_argument("--precheck-atol", type=float, default=2e-5)
    ap.add_argument("--precheck-rtol", type=float, default=2e-5)
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
    out_dir = exp / f"onnx_fp32_fold2_external_compact_{run_tag}"
    onnx_path = out_dir / "heatstroke_tcn_reg_fold2_fp32_external_compact.onnx"

    print("===== FOLD-2 TCN FP32 ONNX EXPORT V2 (EXTERNAL COMPACTION) =====")
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
        raise FileExistsError(f"Refusing to overwrite existing export directory: {out_dir}")

    import onnx
    import onnxruntime as ort
    print("ONNX:", onnx.__version__)
    print("ONNX Runtime:", ort.__version__)

    out_dir.mkdir(parents=True, exist_ok=False)
    helper = import_helper(helper_path)
    project_run = helper.import_project(project)
    dataset = helper.ImputationDataset(project_run, data_dir, "test", args.seed, args.group_column)
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
    print("Valid sensor-value counts:")
    for index in indices:
        print(f"  sample {index}: {sensor_valid_counts(sample_inputs[index][0])}")

    # First prove that the export-safe graph path matches the training model.
    print("\n===== PRE-EXPORT CHECK: ORIGINAL PYTORCH VS EXTERNAL-COMPACT WRAPPER =====")
    precheck_rows = []
    for index in indices:
        x, m, w, compact = sample_inputs[index]
        with torch.no_grad():
            ref = model(x, x_mark_enc=m, one_hot_x=w).cpu().numpy()
            got = wrapper(x, m, w, compact).cpu().numpy()
        metrics = compare_arrays(ref, got, args.precheck_atol, args.precheck_rtol)
        metrics.update({"dataset_index": int(index), "valid_sensor_value_counts": sensor_valid_counts(x)})
        precheck_rows.append(metrics)
        print("PRECHECK:", json.dumps(metrics, sort_keys=True))
        if not metrics["allclose"] or not metrics["finite"]:
            raise RuntimeError(
                f"External-compaction wrapper does not reproduce the original PyTorch model at sample {index}; "
                "do not export this graph."
            )

    trace_index = indices[0]
    first_inputs = sample_inputs[trace_index]
    with torch.no_grad():
        eager_probe = wrapper(*first_inputs)
    if tuple(eager_probe.shape) != (1, 900, 6) or not torch.isfinite(eager_probe).all():
        raise RuntimeError(f"Invalid wrapper probe: shape={tuple(eager_probe.shape)}")
    print("ONNX trace sample index:", trace_index)

    export_kwargs = dict(
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["x_values", "x_mark", "worker_features_18", "compact_values"],
        output_names=["reconstructed_values"],
        dynamic_axes=None,
        verbose=False,
    )
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    print("\nExporting fixed-shape FP32 ONNX with compact_values as explicit input...")
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
        dims = [d.dim_value if d.dim_value else (d.dim_param or "?")
                for d in value.type.tensor_type.shape.dim]
        print(f"Graph tensor {value.name}: dtype={value.type.tensor_type.elem_type}, shape={dims}")

    print("\nCreating ONNX Runtime CPU session...")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    print("ORT inputs:", [(v.name, v.shape, v.type) for v in session.get_inputs()])
    print("ORT outputs:", [(v.name, v.shape, v.type) for v in session.get_outputs()])

    sample_metrics = []
    all_pass = True
    for index in indices:
        x, m, w, compact = sample_inputs[index]
        with torch.no_grad():
            torch_out = model(x, x_mark_enc=m, one_hot_x=w).cpu().numpy()
        feed = {
            "x_values": x.numpy(),
            "x_mark": m.numpy(),
            "worker_features_18": w.numpy(),
            "compact_values": compact.numpy(),
        }
        ort_out = session.run(["reconstructed_values"], feed)[0]
        if torch_out.shape != ort_out.shape:
            raise RuntimeError(
                f"Output shape mismatch at sample {index}: PyTorch={torch_out.shape}, ONNX={ort_out.shape}"
            )
        metrics = compare_arrays(torch_out, ort_out, args.atol, args.rtol)
        metrics.update({
            "dataset_index": int(index),
            "valid_sensor_value_counts": sensor_valid_counts(x),
            "output_shape": list(ort_out.shape),
            "torch_output_min": float(torch_out.min()),
            "torch_output_max": float(torch_out.max()),
            "onnx_output_min": float(ort_out.min()),
            "onnx_output_max": float(ort_out.max()),
            "pass": bool(metrics["allclose"] and metrics["finite"]),
        })
        sample_metrics.append(metrics)
        all_pass = all_pass and metrics["pass"]
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
            "compact_values": [1, 1800, 6],
        },
        "output_spec": {"reconstructed_values": [1, 900, 6]},
        "external_preprocessing": {
            "operation": "per-sensor removal of -9997/-9998/-9999 values followed by adaptive_avg_pool1d to 1800",
            "empty_sensor_result": "all zeros",
            "note": "Firmware must generate compact_values with PyTorch adaptive_avg_pool1d-equivalent bin boundaries.",
        },
        "tcn": {"channels": 32, "kernel_size": 31, "dilations": [1, 2] * 5,
                "theoretical_receptive_field": 901},
        "pre_export_equivalence": {
            "atol": args.precheck_atol,
            "rtol": args.precheck_rtol,
            "all_samples_pass": all(r["allclose"] and r["finite"] for r in precheck_rows),
            "samples": precheck_rows,
        },
        "verification": {
            "runtime": "ONNX Runtime CPUExecutionProvider",
            "trace_dataset_index": int(trace_index),
            "atol": args.atol,
            "rtol": args.rtol,
            "all_samples_pass": all_pass,
            "samples": sample_metrics,
        },
        "scope_note": (
            "Regression/imputation stage only; existing Cla_Model is not included. "
            "No quantization, Vela compilation, firmware build/programming, or flash writes were performed."
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
    print("IMPORTANT: compact_values is a fourth model input and needs a firmware preprocessor.")
    print("No classifier, quantization, Vela, firmware, or flash actions performed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("\n<EXPORT_RESULT> FAIL </EXPORT_RESULT>", file=sys.stderr)
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
