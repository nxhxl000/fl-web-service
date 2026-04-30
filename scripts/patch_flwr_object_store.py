#!/usr/bin/env python3
"""Monkey-patch flwr 1.28's `SqlObjectStore.preregister` to wrap the entire
loop in a single transaction.

Why:
  Without this patch, `preregister` opens a fresh transaction per object in
  the tree. Between those transactions, `confirm_message_received` (called
  from another thread when an unrelated message is consumed) can invoke
  `store.delete()`, which deletes any object with ref_count=0. This includes
  child objects that were just inserted by `preregister` but haven't yet had
  their ref_count bumped — they sit at ref_count=0 in the gap. The race
  causes a `FOREIGN KEY constraint failed` on the next `INSERT INTO
  object_children`, which `flwr-supernode` reports as `Sent successfully`
  but the message is silently lost. The serverapp then waits the full
  `Strategy.start(timeout=3600)` window before dropping the missing reply
  as a straggler, hanging the round for ~60 minutes.

  Wrapping the loop in one outer transaction makes SQLite hold a write
  lock from the first INSERT to the final COMMIT, serializing any concurrent
  delete() calls.

Patch is idempotent: running twice is a no-op (sentinel marker check).
Run from repo root: `python scripts/patch_flwr_object_store.py [--dry-run]`.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

SENTINEL = "# fl-web-service: patched preregister race fix"

UNPATCHED_BLOCK = '''    def preregister(self, run_id: int, object_tree: ObjectTree) -> list[str]:
        """Identify and preregister missing objects in the `ObjectStore`."""
        new_objects = []
        for tree_node in iterate_object_tree(object_tree):
            obj_id = tree_node.object_id
            if not is_valid_sha256_hash(obj_id):
                raise ValueError(f"Invalid object ID format: {obj_id}")

            child_ids = [child.object_id for child in tree_node.children]
            with self.session():
                # Insert new object if it doesn't exist (race-condition safe)
                # RETURNING returns a row only if the insert succeeded
                rows = self.query(
                    "INSERT INTO objects "
                    "(object_id, content, is_available, ref_count) "
                    "VALUES (:object_id, :content, :is_available, :ref_count) "
                    "ON CONFLICT (object_id) DO NOTHING "
                    "RETURNING object_id",
                    {
                        "object_id": obj_id,
                        "content": b"",
                        "is_available": 0,
                        "ref_count": 0,
                    },
                )

                if rows:
                    # New object inserted: set up child relationships
                    for cid in child_ids:
                        self.query(
                            "INSERT INTO object_children (parent_id, child_id) "
                            "VALUES (:parent_id, :child_id)",
                            {"parent_id": obj_id, "child_id": cid},
                        )
                        self.query(
                            "UPDATE objects SET ref_count = ref_count + 1 "
                            "WHERE object_id = :object_id",
                            {"object_id": cid},
                        )
                    new_objects.append(obj_id)
                else:
                    # Object exists: check if unavailable
                    rows = self.query(
                        "SELECT is_available FROM objects WHERE object_id = :object_id",
                        {"object_id": obj_id},
                    )
                    if rows and not rows[0]["is_available"]:
                        new_objects.append(obj_id)

                # Ensure run mapping
                self.query(
                    "INSERT INTO run_objects (run_id, object_id) "
                    "VALUES (:run_id, :object_id) ON CONFLICT DO NOTHING",
                    {"run_id": uint64_to_int64(run_id), "object_id": obj_id},
                )
        return new_objects'''

PATCHED_BLOCK = f'''    def preregister(self, run_id: int, object_tree: ObjectTree) -> list[str]:
        """Identify and preregister missing objects in the `ObjectStore`.

        {SENTINEL}
        Loop wrapped in ONE outer `with self.session()` so SQLite holds an
        exclusive write lock end-to-end. This serializes concurrent
        `store.delete()` (called by `confirm_message_received` on every
        consumed message), which would otherwise wipe a freshly-inserted
        child with ref_count=0 between iterations and crash the next
        `INSERT INTO object_children` on FK violation.
        """
        new_objects = []
        with self.session():
            for tree_node in iterate_object_tree(object_tree):
                obj_id = tree_node.object_id
                if not is_valid_sha256_hash(obj_id):
                    raise ValueError(f"Invalid object ID format: {{obj_id}}")

                child_ids = [child.object_id for child in tree_node.children]
                # Insert new object if it doesn't exist (race-condition safe)
                # RETURNING returns a row only if the insert succeeded
                rows = self.query(
                    "INSERT INTO objects "
                    "(object_id, content, is_available, ref_count) "
                    "VALUES (:object_id, :content, :is_available, :ref_count) "
                    "ON CONFLICT (object_id) DO NOTHING "
                    "RETURNING object_id",
                    {{
                        "object_id": obj_id,
                        "content": b"",
                        "is_available": 0,
                        "ref_count": 0,
                    }},
                )

                if rows:
                    # New object inserted: set up child relationships
                    for cid in child_ids:
                        self.query(
                            "INSERT INTO object_children (parent_id, child_id) "
                            "VALUES (:parent_id, :child_id)",
                            {{"parent_id": obj_id, "child_id": cid}},
                        )
                        self.query(
                            "UPDATE objects SET ref_count = ref_count + 1 "
                            "WHERE object_id = :object_id",
                            {{"object_id": cid}},
                        )
                    new_objects.append(obj_id)
                else:
                    # Object exists: check if unavailable
                    rows = self.query(
                        "SELECT is_available FROM objects WHERE object_id = :object_id",
                        {{"object_id": obj_id}},
                    )
                    if rows and not rows[0]["is_available"]:
                        new_objects.append(obj_id)

                # Ensure run mapping
                self.query(
                    "INSERT INTO run_objects (run_id, object_id) "
                    "VALUES (:run_id, :object_id) ON CONFLICT DO NOTHING",
                    {{"run_id": uint64_to_int64(run_id), "object_id": obj_id}},
                )
        return new_objects'''


def find_target() -> Path:
    """Locate flwr's sql_object_store.py inside whichever venv is active."""
    spec = importlib.util.find_spec("flwr.supercore.object_store.sql_object_store")
    if spec is None or spec.origin is None:
        raise RuntimeError(
            "Could not locate `flwr.supercore.object_store.sql_object_store`. "
            "Activate the venv that has flwr installed and try again."
        )
    return Path(spec.origin)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="show diff but don't write")
    parser.add_argument("--revert", action="store_true", help="restore original (drops patch)")
    args = parser.parse_args()

    target = find_target()
    src = target.read_text()

    already_patched = SENTINEL in src

    if args.revert:
        if not already_patched:
            print(f"[noop] not patched: {target}")
            return 0
        if PATCHED_BLOCK not in src:
            print(f"[error] patch marker present but block doesn't match — manual revert needed", file=sys.stderr)
            return 2
        new_src = src.replace(PATCHED_BLOCK, UNPATCHED_BLOCK)
        if args.dry_run:
            print(f"[dry-run] would revert: {target}")
            return 0
        target.write_text(new_src)
        print(f"[ok] reverted: {target}")
        return 0

    if already_patched:
        print(f"[noop] already patched: {target}")
        return 0

    if UNPATCHED_BLOCK not in src:
        print(
            f"[error] target has unexpected content; cannot patch automatically.\n"
            f"  file: {target}\n"
            f"  hint: flwr version may have changed. Check upstream and update this script.",
            file=sys.stderr,
        )
        return 2

    new_src = src.replace(UNPATCHED_BLOCK, PATCHED_BLOCK)
    if args.dry_run:
        print(f"[dry-run] would patch: {target}")
        print(f"[dry-run] (delta: {len(new_src) - len(src):+d} chars)")
        return 0

    target.write_text(new_src)
    print(f"[ok] patched: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
