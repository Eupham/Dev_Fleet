"""Runtime monkey-patch for Chainlit's load_user_env.

In some Chainlit 2.x versions, load_user_env raises UnboundLocalError when
the browser sends userEnv: null (i.e. when CHAINLIT_USER_ENV is not set).
This module patches the function at runtime to return {} in that case.

Import this module early in your application (before chainlit is used) to apply the patch.
"""

import sys


def apply_patch() -> None:
    """Patch chainlit.socket.load_user_env to handle missing userEnv gracefully."""
    try:
        import importlib
        import chainlit.socket as _socket_module

        src = None
        # Try to get the source file path for the build-time approach
        import inspect
        try:
            src_file = inspect.getfile(_socket_module)
        except (TypeError, OSError):
            src_file = None

        # Runtime monkey-patch: wrap load_user_env to never raise UnboundLocalError
        original_fn = getattr(_socket_module, "load_user_env", None)
        if original_fn is None:
            return

        def _safe_load_user_env(*args, **kwargs):
            try:
                result = original_fn(*args, **kwargs)
                return result if result is not None else {}
            except (UnboundLocalError, AttributeError, KeyError):
                return {}

        _socket_module.load_user_env = _safe_load_user_env
        print("[fix_chainlit] Runtime monkey-patch applied to load_user_env OK")

    except ImportError:
        # Chainlit not installed — nothing to patch
        pass
    except Exception as exc:
        print(f"[fix_chainlit] Patch failed ({exc}) — continuing anyway", file=sys.stderr)


# Apply patch immediately when this module is imported
apply_patch()
