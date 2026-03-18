import sys
import struct

def fix_arch(filepath):
    print("Fixing GGUF architecture string in", filepath)
    with open(filepath, 'r+b') as f:
        f.seek(0)
        chunk = f.read(1000000)

        search_pattern = b'\x14\x00\x00\x00\x00\x00\x00\x00general.architecture\x08\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00\x00qwen35'
        idx = chunk.find(search_pattern)
        if idx == -1:
            print("Could not find the general.architecture 'qwen35' pattern.")
            sys.exit(1)

        print("Found pattern at offset", idx)

        # Rewrite the file, skipping the '35' part and changing length from 6 to 5.

        # Read the entire file... 2.5GB. We will write to a new file to be safe!
        # wait, if I write to a new file, it's safer and easier.
        pass

def fix_arch_new(filepath, outpath):
    print(f"Fixing GGUF {filepath} -> {outpath}")
    with open(filepath, 'rb') as f, open(outpath, 'wb') as out:
        chunk = f.read(100000)
        search_pattern = b'\x14\x00\x00\x00\x00\x00\x00\x00general.architecture\x08\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00\x00qwen35'
        idx = chunk.find(search_pattern)
        if idx == -1:
            print("Not found")
            return

        print("Found at", idx)

        # Write up to the length
        len_offset = idx + 20 + 4
        out.write(chunk[:len_offset])
        out.write(struct.pack('<Q', 5))
        out.write(b'qwen2')

        # Now write the rest
        rest_offset = len_offset + 8 + 6
        out.write(chunk[rest_offset:])

        while True:
            data = f.read(1024*1024*16)
            if not data:
                break
            out.write(data)
    print("Done")

if __name__ == "__main__":
    if len(sys.argv) > 2:
        fix_arch_new(sys.argv[1], sys.argv[2])
