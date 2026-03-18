import sys
import struct

def fix_arch_new(filepath, outpath):
    print(f"Fixing GGUF {filepath} -> {outpath}")
    with open(filepath, 'rb') as f, open(outpath, 'wb') as out:
        chunk = f.read(100000)
        idx = chunk.find(b'general.architecture')
        if idx == -1:
            print("Not found")
            return

        print("Found at", idx)

        # Read the type and length
        vtype_offset = idx + len(b'general.architecture')
        vtype = struct.unpack('<I', chunk[vtype_offset:vtype_offset+4])[0]
        if vtype == 8:
            vlen_offset = vtype_offset + 4
            vlen = struct.unpack('<Q', chunk[vlen_offset:vlen_offset+8])[0]
            val = chunk[vlen_offset+8:vlen_offset+8+vlen]
            print(f"Value string: {val}")

            if val == b'qwen35':
                out.write(chunk[:vlen_offset])
                # We can't change the length to 5 without misaligning the rest of the file
                # GGUF tensors might require specific alignments.
                # So we must pad the string to 6 bytes. But how?
                # GGUF strings don't have to be null-terminated.
                # wait! If we pad it to `qwen2 ` it's 6 bytes.
                # Or wait... GGUF might read exactly 6 bytes.
                pass
