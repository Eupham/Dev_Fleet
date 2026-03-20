"""Dev Fleet Volume Manager — sync the repo to/from a Modal Volume.

This enables volume-based deployment: the full source tree lives in
``dev-fleet-repo`` (a Modal Volume) so deployment never needs a git clone.

Second-set credentials
-----------------------
Set the env vars below *before* running push/pull so this script uses the
dedicated volume account instead of the primary deployment account:

    export MODAL_TOKEN_ID="$MODAL_VOLUME_TOKEN_ID"
    export MODAL_TOKEN_SECRET="$MODAL_VOLUME_TOKEN_SECRET"

Or pass them inline:

    MODAL_TOKEN_ID=xxx MODAL_TOKEN_SECRET=yyy python volume_manager.py push

Commands
---------
    python volume_manager.py push          # upload local repo → volume
    python volume_manager.py push --dry-run
    python volume_manager.py pull          # download volume → /tmp/dev-fleet-deploy
    python volume_manager.py pull --dest /path/to/dir
    python volume_manager.py ls            # list files in the volume
    python volume_manager.py ls /subdir
"""

import argparse
import fnmatch
import pathlib
import sys

import modal

# ── constants ────────────────────────────────────────────────────────────────

VOLUME_NAME = "dev-fleet-repo"
REPO_ROOT = pathlib.Path(__file__).parent.resolve()
DEFAULT_PULL_DEST = "/tmp/dev-fleet-deploy"

# Patterns matched against any path *component* (not full path).
# Mirrors .gitignore + adds Modal-specific junk.
_EXCLUDE_PATTERNS = [
    ".git",
    ".modal",
    "__pycache__",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "*.egg-info",
    ".pytest_cache",
    ".coverage",
    "venv",
    ".venv",
    "node_modules",
    "*.log",
    ".env",
    "auto_deploy.py",
    "payload.zip",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _excluded(path: pathlib.Path) -> bool:
    """Return True if *any* component of path matches an exclude pattern."""
    for part in path.parts:
        for pat in _EXCLUDE_PATTERNS:
            if fnmatch.fnmatch(part, pat):
                return True
    return False


def _iter_repo_files(root: pathlib.Path):
    """Yield (local_path, remote_path) for every file to upload."""
    for src in sorted(root.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(root)
        if _excluded(rel):
            continue
        yield src, f"/{rel}"


# ── commands ──────────────────────────────────────────────────────────────────

def cmd_push(dry_run: bool = False) -> None:
    """Upload the local repo tree to the Modal volume."""
    vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

    files = list(_iter_repo_files(REPO_ROOT))
    print(f"{'[dry-run] ' if dry_run else ''}Pushing {len(files)} files → volume '{VOLUME_NAME}' …")

    if dry_run:
        for _, remote in files:
            print(f"  {remote}")
        return

    with vol.batch_upload(force=True) as batch:
        for local, remote in files:
            batch.put_file(str(local), remote)
            print(f"  ↑ {remote}")

    print(f"✓ Push complete — {len(files)} files in '{VOLUME_NAME}'")


def cmd_pull(dest: str = DEFAULT_PULL_DEST) -> None:
    """Download the volume to *dest* (ready for `modal deploy`)."""
    dest_path = pathlib.Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)

    vol = modal.Volume.from_name(VOLUME_NAME)

    count = 0
    for entry in vol.listdir("/", recursive=True):
        if entry.type != modal.volume.FileEntryType.FILE:
            continue
        local_file = dest_path / entry.path.lstrip("/")
        local_file.parent.mkdir(parents=True, exist_ok=True)
        data = b"".join(vol.read_file(entry.path))
        local_file.write_bytes(data)
        print(f"  ↓ {entry.path}")
        count += 1

    print(f"✓ Pull complete — {count} files written to {dest}")


def cmd_ls(remote_dir: str = "/") -> None:
    """List files inside the volume."""
    vol = modal.Volume.from_name(VOLUME_NAME)
    entries = list(vol.listdir(remote_dir, recursive=True))
    if not entries:
        print(f"(volume '{VOLUME_NAME}' is empty or does not exist)")
        return
    for e in entries:
        kind = "d" if e.type == modal.volume.FileEntryType.DIRECTORY else "f"
        print(f"  [{kind}] {e.path}")
    print(f"\n{len(entries)} entries in '{VOLUME_NAME}:{remote_dir}'")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dev Fleet Volume Manager — repo ↔ Modal Volume sync",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    push_p = sub.add_parser("push", help="Upload local repo to Modal volume")
    push_p.add_argument(
        "--dry-run", action="store_true",
        help="Print files that would be uploaded without uploading",
    )

    pull_p = sub.add_parser("pull", help="Download Modal volume to local directory")
    pull_p.add_argument(
        "--dest", default=DEFAULT_PULL_DEST,
        help=f"Destination directory (default: {DEFAULT_PULL_DEST})",
    )

    ls_p = sub.add_parser("ls", help="List files in the Modal volume")
    ls_p.add_argument("path", nargs="?", default="/", help="Remote path to list")

    args = parser.parse_args()

    if args.cmd == "push":
        cmd_push(dry_run=args.dry_run)
    elif args.cmd == "pull":
        cmd_pull(dest=args.dest)
    elif args.cmd == "ls":
        cmd_ls(remote_dir=args.path)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
