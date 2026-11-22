import os
import glob
import hashlib
import traceback
import numpy as np

from tensorflow.lite.python import schema_py_generated as schema_fb

MODEL_DIR = os.environ["MODEL_DIR"]
OSPI_BASE = 0xC0000000

def audit_model(path):
    print("\n" + "=" * 100)
    print("MODEL:", os.path.basename(path))
    print("FILE_BYTES:", os.path.getsize(path))

    with open(path, "rb") as f:
        raw = bytearray(f.read())

    print("SHA256:", hashlib.sha256(raw).hexdigest())

    host_base = np.frombuffer(raw, dtype=np.uint8)
    model = schema_fb.Model.GetRootAsModel(raw, 0)

    total = 0
    bad = 0

    for sg_idx in range(model.SubgraphsLength()):
        sg = model.Subgraphs(sg_idx)

        for op_idx in range(sg.OperatorsLength()):
            op = sg.Operators(op_idx)
            opcode = model.OperatorCodes(op.OpcodeIndex())
            custom_code = opcode.CustomCode()

            if isinstance(custom_code, bytes):
                custom_code = custom_code.decode("utf-8", errors="replace")

            if custom_code != "ethos-u":
                continue

            total += 1

            if op.InputsLength() < 1:
                print("ERROR: Ethos-U op has no input 0:", op_idx)
                bad += 1
                continue

            cmd_tensor_idx = int(op.Inputs(0))

            if cmd_tensor_idx < 0:
                print("ERROR: Invalid command-stream tensor:", op_idx)
                bad += 1
                continue

            tensor = sg.Tensors(cmd_tensor_idx)
            buffer_idx = int(tensor.Buffer())
            tensor_name = tensor.Name()

            if isinstance(tensor_name, bytes):
                tensor_name = tensor_name.decode("utf-8", errors="replace")

            buf = model.Buffers(buffer_idx)
            data = buf.DataAsNumpy()

            if data is None or len(data) == 0:
                print(
                    f"ERROR: Empty command stream: op={op_idx}, "
                    f"tensor={cmd_tensor_idx}, buffer={buffer_idx}"
                )
                bad += 1
                continue

            if not np.shares_memory(data, host_base):
                print(
                    "ERROR: Cannot establish FlatBuffer data offset:",
                    "op=", op_idx,
                    "tensor=", cmd_tensor_idx,
                    "buffer=", buffer_idx
                )
                bad += 1
                continue

            file_offset = int(data.ctypes.data - host_base.ctypes.data)
            target_addr = OSPI_BASE + file_offset
            remainder = target_addr % 16
            aligned = remainder == 0

            print(
                f"ETHOSU_OP={op_idx} "
                f"CMD_TENSOR={cmd_tensor_idx} "
                f"TENSOR_NAME={tensor_name} "
                f"BUFFER={buffer_idx} "
                f"BYTES={len(data)}"
            )
            print(
                f"  FILE_OFFSET=0x{file_offset:X} "
                f"TARGET_ADDR=0x{target_addr:X} "
                f"MOD16={remainder} "
                f"ALIGNED={'YES' if aligned else 'NO'}"
            )

            if not aligned:
                bad += 1

    print("ETHOSU_CUSTOM_NODE_COUNT:", total)
    print("MISALIGNED_OR_INVALID_COMMAND_STREAMS:", bad)
    print("COMMAND_STREAM_ALIGNMENT_STATUS:",
          "PASS" if total > 0 and bad == 0 else "FAIL")

    return total, bad


pattern = os.path.join(
    MODEL_DIR,
    "heatstroke_tcn_reg_fold2_gemm_only_qdq_vela*.tflite"
)

paths = sorted(glob.glob(pattern))

print("MODEL_DIR:", MODEL_DIR)
print("OSPI_BASE:", hex(OSPI_BASE))
print("MODELS_FOUND:", len(paths))

if not paths:
    print("ERROR: No matching model files found.")
else:
    all_bad = 0

    for path in paths:
        try:
            _, bad = audit_model(path)
            all_bad += bad
        except Exception:
            print("MODEL_AUDIT_EXCEPTION:", path)
            traceback.print_exc()

    print("\n" + "=" * 100)
    print("TOTAL_ALIGNMENT_ERRORS_ACROSS_AUDITED_MODELS:", all_bad)
    print("AUDIT_FINISHED")
