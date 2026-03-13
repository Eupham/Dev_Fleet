"""
Patch Chainlit's load_user_env to handle missing userEnv gracefully.

In some Chainlit 2.x versions, load_user_env raises UnboundLocalError when
the browser sends userEnv: null (i.e. when CHAINLIT_USER_ENV is not set).
This script patches socket.py at image build time to return {} in that case.
"""
import pathlib

TARGET = pathlib.Path('/usr/local/lib/python3.12/site-packages/chainlit/socket.py')

if not TARGET.exists():
    print(f"[fix_chainlit] {TARGET} not found — skipping")
else:
    src = TARGET.read_text()
    patched = src.replace(
        '    return user_env_dict\n',
        '    return user_env_dict if "user_env_dict" in locals() else {}\n',
        1,
    )
    if patched != src:
        TARGET.write_text(patched)
        print("[fix_chainlit] Patched load_user_env OK")
    else:
        print("[fix_chainlit] Pattern not found — already fixed or unexpected version")
