import struct
import json
import sys

ESMF_MAGIC = b"ESMF"
ENTRY_FMT = "<64sII4qQQ"
ENTRY_SIZE = struct.calcsize(ENTRY_FMT)

def print_tensors(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != ESMF_MAGIC:
            print("Not a valid ESMF file")
            return
        version = struct.unpack("<I", f.read(4))[0]
        meta_size = struct.unpack("<Q", f.read(8))[0]
        f.read(meta_size) # Skip metadata
        n_tensors = struct.unpack("<I", f.read(4))[0]
        
        for _ in range(n_tensors):
            raw = f.read(ENTRY_SIZE)
            name_b = struct.unpack(ENTRY_FMT, raw)[0]
            name = name_b.rstrip(b"\x00").decode("utf-8")
            print(name)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print_tensors(sys.argv[1])
    else:
        print("Usage: python print_tensors.py <file.esmf>")
