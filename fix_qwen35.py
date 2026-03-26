import sys

def fix_arch(filepath):
    print("Fixing", filepath)
    with open(filepath, 'r+b') as f:
        content = f.read(2000000)
        idx = content.find(b'qwen35')
        while idx != -1:
            print(f'Found qwen35 at offset {idx}')
            f.seek(idx)
            f.write(b'qwen2 ')
            idx = content.find(b'qwen35', idx+1)

fix_arch('test_model.gguf')
