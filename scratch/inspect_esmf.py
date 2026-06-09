import struct
import json

path = "/home/ace/gitfiles/ESMF/esmc_300m.esmf"
ENTRY_FMT = "<64sII4qQQ"
ENTRY_SIZE = struct.calcsize(ENTRY_FMT)

with open(path, "rb") as f:
    magic = f.read(4)
    version = struct.unpack("<I", f.read(4))[0]
    meta_size = struct.unpack("<Q", f.read(8))[0]
    meta_json = json.loads(f.read(meta_size).decode("utf-8"))
    print("Metadata:", meta_json)
    n_tensors = struct.unpack("<I", f.read(4))[0]
    for _ in range(n_tensors):
        raw = f.read(ENTRY_SIZE)
        name_b, dtype, ndim, *shape_and_offsets = struct.unpack(ENTRY_FMT, raw)
        shape = shape_and_offsets[:ndim]
        name = name_b.rstrip(b"\x00").decode("utf-8")
        if "ffn" in name or "embd" in name:
            print(f"{name:40s} {shape}")
