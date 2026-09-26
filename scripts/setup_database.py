"""
Set up PostgreSQL for GridSentinel on this machine (no Docker, no psql needed).

    python scripts/setup_database.py
    python scripts/setup_database.py --host localhost --port 5432 --admin-user postgres

Connects as an administrator (asks for the password, or reads PGPASSWORD) and, if
they do not exist yet, creates:

- the login role ``gridsentinel`` (its password is asked, Enter reuses the admin one;
  with PGPASSWORD set, that password is used for both);
- the database ``gridsentinel``, owned by that role, for the console;
- the database ``gridsentinel_test``, for ``pytest``, so tests never touch the console's data.

Then it creates the tables and writes ``DATABASE_URL`` and ``TEST_DATABASE_URL`` into
``.env`` in the repository root (gitignored; other lines are kept), which the API and
the tests read. Running it again is safe: existing objects are kept and the role's
password is updated. ``--no-write-env`` prints the two lines instead.
"""

import argparse
import getpass
import os
import sys
from pathlib import Path
from urllib.parse import quote

import psycopg
from psycopg import sql
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services import db  # noqa: E402  (the engine is built from DATABASE_URL on first use)


def connection_url(user: str, password: str, host: str, port: int, database: str) -> str:
    return f"postgresql://{quote(user, safe='')}:{quote(password, safe='')}@{host}:{port}/{database}"


def ensure_role(admin, role: str, password: str) -> str:
    exists = admin.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role,)).fetchone()
    verb = "ALTER" if exists else "CREATE"
    admin.execute(sql.SQL(verb + " ROLE {} LOGIN PASSWORD {}").format(sql.Identifier(role), sql.Literal(password)))
    return f"role {role}: {'password updated' if exists else 'created'}"


def ensure_database(admin, database: str, owner: str) -> str:
    if admin.execute("SELECT 1 FROM pg_database WHERE datname = %s", (database,)).fetchone():
        return f"database {database}: exists"
    # Owned by the app role, so it may create tables in "public" (PostgreSQL 15+). UTF-8 from template0,
    # which allows any encoding; if the server's locale rules UTF-8 out, keep the server's default.
    name, role = sql.Identifier(database), sql.Identifier(owner)
    try:
        admin.execute(sql.SQL("CREATE DATABASE {} OWNER {} TEMPLATE template0 ENCODING 'UTF8'").format(name, role))
    except psycopg.errors.InvalidParameterValue:
        admin.execute(sql.SQL("CREATE DATABASE {} OWNER {}").format(name, role))
    encoding = admin.execute("SELECT pg_encoding_to_char(encoding)::text FROM pg_database WHERE datname = %s", (database,)).fetchone()[0]
    if isinstance(encoding, bytes):  # a SQL_ASCII server returns undecoded text
        encoding = encoding.decode("ascii", "replace")
    return f"database {database}: created ({encoding})"


RESET_STEPS = r"""
To reset a forgotten postgres password on Windows (PostgreSQL 18; adjust the version):
  1. Open C:\Program Files\PostgreSQL\18\data\pg_hba.conf as administrator and change
     scram-sha-256 to trust on the lines for 127.0.0.1/32 and ::1/128.
  2. Restart the service: services.msc > postgresql-x64-18 > Restart.
  3. "C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -h localhost -c "ALTER USER postgres PASSWORD 'new-password'"
  4. Change trust back to scram-sha-256 and restart the service again."""


def server_answers(host: str, port: int) -> bool:
    """True if a PostgreSQL server listens on host:port (a login is not needed to tell)."""
    try:
        psycopg.connect(host=host, port=port, user="postgres", dbname="postgres", connect_timeout=3).close()
        return True
    except psycopg.OperationalError as exc:
        text = str(exc).lower()
        return any(word in text for word in ("password", "authentication", "pg_hba", "does not exist"))


def explain_failure(error: str, args, password_from_env: bool) -> None:
    """Tell apart a wrong password, a different server on this port and a stopped service."""
    text = error.lower()
    others = [port for port in range(args.port + 1, args.port + 4) if server_answers(args.host, port)]
    elsewhere = (f"A PostgreSQL server also answers on port {', '.join(map(str, others))}. If that is the one you "
                 f"installed, run: python scripts/setup_database.py --port {others[0]}") if others else None
    if "password authentication failed" in text or "no password supplied" in text:
        print(f"\nThe server on port {args.port} rejected the password for {args.admin_user!r}.")
        if elsewhere:
            print(elsewhere)
        else:
            print(f"No other PostgreSQL server answers on ports {args.port + 1}-{args.port + 3}, so the password is the likely problem.")
        if password_from_env:
            print("The password came from the PGPASSWORD variable; clear it (set PGPASSWORD=) to be asked instead.")
        else:
            print("On Windows, Ctrl+V does not paste into the hidden prompt. To skip the prompt, run:")
            print("  set PGPASSWORD=<your postgres password>")
            print("  python scripts/setup_database.py")
        print(RESET_STEPS)
    elif "refused" in text or "timeout" in text or "timed out" in text:
        print(f"\nNothing accepts connections on {args.host}:{args.port}: start the PostgreSQL service "
              "(services.msc > postgresql-x64-18 > Start).")
        if elsewhere:
            print(elsewhere)
    else:
        print("Check that the PostgreSQL service is running and the password is right.")


def write_env(values: dict) -> Path:
    """Set ``values`` in .env, replacing those keys and keeping every other line."""
    path = ROOT / ".env"
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else [
        "# Local settings for GridSentinel (not committed). See .env.example for the others.",
        "ENV=development",
    ]
    kept = [line for line in lines if line.split("=", 1)[0].strip() not in values]
    path.write_text("\n".join(kept + [f"{key}={value}" for key, value in values.items()]) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Create the GridSentinel PostgreSQL role and databases")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--admin-user", default="postgres", help="an existing superuser (default: postgres)")
    parser.add_argument("--role", default="gridsentinel", help="login role the API uses")
    parser.add_argument("--database", default="gridsentinel")
    parser.add_argument("--test-database", default="gridsentinel_test")
    parser.add_argument("--no-write-env", action="store_true", help="print the settings instead of writing .env")
    args = parser.parse_args()

    from_env = bool(os.getenv("PGPASSWORD"))
    admin_password = os.getenv("PGPASSWORD") or getpass.getpass(f"Password of PostgreSQL user {args.admin_user!r}: ")

    try:
        admin = psycopg.connect(host=args.host, port=args.port, user=args.admin_user, password=admin_password,
                                dbname="postgres", autocommit=True, connect_timeout=10)
    except psycopg.OperationalError as exc:
        print(f"Could not connect to PostgreSQL at {args.host}:{args.port} as {args.admin_user}: {exc}".strip())
        explain_failure(str(exc), args, password_from_env=from_env)
        return 1
    with admin:
        print(f"Connected: {admin.execute('SHOW server_version').fetchone()[0]}")
        # With PGPASSWORD set the app role gets the same password; otherwise it is asked for.
        role_password = admin_password if from_env else (
            getpass.getpass(f"Password for the {args.role!r} role (Enter: the same): ") or admin_password)
        print(ensure_role(admin, args.role, role_password))
        for database in (args.database, args.test_database):
            print(ensure_database(admin, database, args.role))

    urls = {
        "DATABASE_URL": connection_url(args.role, role_password, args.host, args.port, args.database),
        "TEST_DATABASE_URL": connection_url(args.role, role_password, args.host, args.port, args.test_database),
    }
    # Create the tables as the app role, exactly as the API does at startup.
    os.environ["DATABASE_URL"] = urls["DATABASE_URL"]
    db.init_db()
    with db.transaction() as connection:
        tables = connection.execute(text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")).scalars().all()
    print(f"tables in {args.database}: {', '.join(sorted(tables))}")

    if args.no_write_env:
        print("\nAdd these lines to .env:")
        for key, value in urls.items():
            print(f"{key}={value}")
    else:
        print(f"\nWrote DATABASE_URL and TEST_DATABASE_URL to {write_env(urls)}")
    print("Next: uvicorn backend.main:app --reload   (then /api/health shows \"database\": \"postgresql\")")
    return 0


if __name__ == "__main__":
    sys.exit(main())
