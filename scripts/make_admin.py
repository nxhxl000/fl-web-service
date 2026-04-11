"""Promote a registered user to admin.

Usage:
    python -m scripts.make_admin user@example.com

Admin is identified by the `is_admin` boolean column on `users`. There is no
registration flow that sets this — run this script from a shell with access
to the database (via `.env` / DATABASE_URL).
"""

import sys

from backend.auth.models import User
from backend.clients import models as _client_models  # noqa: F401 — register mapper
from backend.db import SessionLocal
from backend.projects import models as _project_models  # noqa: F401 — register mapper


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: python -m scripts.make_admin <email>", file=sys.stderr)
        return 2

    email = sys.argv[1]
    with SessionLocal() as db:
        user = db.query(User).filter(User.email == email).one_or_none()
        if user is None:
            print(f"user not found: {email}", file=sys.stderr)
            return 1
        if user.is_admin:
            print(f"{email} is already admin")
            return 0
        user.is_admin = True
        db.commit()
        print(f"{email} is now admin")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
