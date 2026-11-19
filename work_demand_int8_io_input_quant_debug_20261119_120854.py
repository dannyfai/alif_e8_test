import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf

OLD_TFLITE = os.environ["OLD_TFLITE"]
NEW_TFLITE = os.environ["NEW_TFLITE"]
VECTORS = os.environ["VECTORS"]

INPUT_SCALE = 0.005934785585850477
INPUT_ZERO_POINT = -84

print("TensorFlow:", tf.__version__)
print("NumPy:", np.__version__)

vectors = np.load(VECTORS)
print("Original vector array:", vectors.shape, vectors.dtype)

if vectors.ndim == 4 and vectors.shape[1:] == (1, 3, 512):
    vectors = vectors[:, 0]
elif vectors.ndim == 3 and vectors.shape[1:] == (3, 512):
    pass
elif vectors.ndim == 2 and vectors.shape == (3, 512):
    vectors = vectors[None, ...]
else:
    raise RuntimeError(f"Unexpected vector shape: {vectors.shape}")

print("Normalized vectors:", vectors.shape)

old = tf.lite.Interpreter(
    model_path=OLD_TFLITE,
    experimental_preserve_all_tensors=True
)
new = tf.lite.Interpreter(
    model_path=NEW_TFLITE,
    experimental_preserve_all_tensors=True
)

old.allocate_tensors()
new.allocate_tensors()

old_in = old.get_input_details()[0]
new_in = new.get_input_details()[0]

print("\n=== Input details ===")
print("Old:", old_in["name"], old_in["shape"], old_in["dtype"])
print("New:", new_in["name"], new_in["shape"], new_in["dtype"])
print("New input quantization:", new_in["quantization"])

old_tensors = old.get_tensor_details()
new_tensors = new.get_tensor_details()

def quant_variants(x):
    x = np.asarray(x, dtype=np.float32)

    # Float32 round-to-nearest-even.
    v32 = x / np.float32(INPUT_SCALE)
    q_rint32 = np.clip(
        np.rint(v32) + INPUT_ZERO_POINT, -128, 127
    ).astype(np.int8)

    # Float32 round-half-away-from-zero.
    rounded_away = np.where(
        v32 >= 0,
        np.floor(v32 + 0.5),
        np.ceil(v32 - 0.5)
    )
    q_away32 = np.clip(
        rounded_away + INPUT_ZERO_POINT, -128, 127
    ).astype(np.int8)

    # Float64 round-to-nearest-even.
    v64 = x.astype(np.float64) / INPUT_SCALE
    q_rint64 = np.clip(
        np.rint(v64) + INPUT_ZERO_POINT, -128, 127
    ).astype(np.int8)

    return {
        "rint_float32": q_rint32,
        "away_float32": q_away32,
        "rint_float64": q_rint64,
    }

def print_candidate_input_tensors(interpreter, details, label, expected):
    print(f"\n--- {label}: input-shaped INT8 intermediate tensors ---")
    found = 0

    for d in details:
        if np.dtype(d["dtype"]) != np.dtype(np.int8):
            continue

        shape = tuple(int(v) for v in d["shape"])
        if shape not in (
            (1, 3, 512),
            (1, 512, 3),
            (1, 512, 1, 3),
        ):
            continue

        try:
            value = interpreter.get_tensor(d["index"])
        except Exception:
            continue

        if shape == (1, 3, 512):
            targets = {
                k: q for k, q in expected.items()
            }
        elif shape == (1, 512, 3):
            targets = {
                k: q.transpose(0, 2, 1)
                for k, q in expected.items()
            }
        else:
            targets = {
                k: q.transpose(0, 2, 1).reshape(1, 512, 1, 3)
                for k, q in expected.items()
            }

        differences = {
            k: int(np.count_nonzero(value != target))
            for k, target in targets.items()
        }

        print(
            f"index={d['index']} name={d['name']!r} "
            f"shape={shape} dtype={d['dtype']} "
            f"min={int(value.min())} max={int(value.max())} "
            f"diff_counts={differences}"
        )
        found += 1

    if found == 0:
        print("[INFO] No matching internal INT8 input tensors found.")

def get_output_by_name(details, stage):
    matches = [
        d for d in details
        if stage in d["name"]
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Could not uniquely find {stage} output: "
            f"{[d['name'] for d in matches]}"
        )
    return matches[0]

stages = ["stage1", "stage2", "stage3a", "stage3b"]
old_outputs = old.get_output_details()
new_outputs = new.get_output_details()

old_out_map = {
    stage: get_output_by_name(old_outputs, stage)
    for stage in stages
}
new_out_map = {
    stage: get_output_by_name(new_outputs, stage)
    for stage in stages
}

print("\n=== Output quantization details ===")
for stage in stages:
    d = new_out_map[stage]
    qp = d.get("quantization_parameters", {})
    scales = qp.get("scales", [])
    zps = qp.get("zero_points", [])
    print(
        stage,
        "name=", d["name"],
        "dtype=", d["dtype"],
        "scale=", scales,
        "zero_point=", zps
    )

print("\n=== Per-vector test ===")

for sample_index, sample in enumerate(vectors):
    x = np.asarray(sample[None, ...], dtype=np.float32)
    variants = quant_variants(x)

    # Feed original FP32 input to the original model.
    old.set_tensor(old_in["index"], x)
    old.invoke()

    # Feed externally quantized INT8 input to the new model.
    q_input = variants["rint_float32"]
    new.set_tensor(new_in["index"], q_input)
    new.invoke()

    print(f"\nSample {sample_index}")
    if sample_index == 0:
        print_candidate_input_tensors(
            old, old_tensors, "OLD MODEL", variants
        )
        print_candidate_input_tensors(
            new, new_tensors, "NEW MODEL", variants
        )

    for stage in stages:
        od = old_out_map[stage]
        nd = new_out_map[stage]

        old_value = old.get_tensor(od["index"]).astype(np.float32)
        new_codes = new.get_tensor(nd["index"]).astype(np.int8)

        qp = nd["quantization_parameters"]
        scale = float(qp["scales"][0])
        zp = int(qp["zero_points"][0])

        # Reconstruct the old model's quantized output codes.
        old_codes = np.clip(
            np.rint(old_value / scale) + zp,
            -128,
            127
        ).astype(np.int8)

        code_diff = (
            new_codes.astype(np.int16) -
            old_codes.astype(np.int16)
        )

        new_dequant = (
            new_codes.astype(np.float32) - zp
        ) * scale

        print(
            f"{stage}: "
            f"old_codes={old_codes.reshape(-1).tolist()} "
            f"new_codes={new_codes.reshape(-1).tolist()} "
            f"code_diff={code_diff.reshape(-1).tolist()} "
            f"max_abs_logit_diff="
            f"{float(np.max(np.abs(old_value-new_dequant))):.9g}"
        )

print("\n=== Diagnostic completed ===")
