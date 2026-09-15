"""Database access for the voucher fact layer.

Credentials are injected at runtime from the environment and are never read
from, written to, or echoed out of ``.env``. Source the project env into the
shell before running any script in this package, or export ``PGPASSWORD``
from the container config.

Recognised variables, in precedence order:

===================  ==========================================
``PGHOST``           ``POSTGRES_HOST``   (default ``127.0.0.1``)
``PGPORT``           ``POSTGRES_PORT``   (default ``5432``)
``PGDATABASE``       ``POSTGRES_DB``     (default ``boarddocs``)
``PGUSER``           ``POSTGRES_USER``   (default ``boarddocs``)
``PGPASSWORD``       ``POSTGRES_PASSWORD``
===================  ==========================================

Always 127.0.0.1, never ``localhost``: Fedora resolves ``localhost`` to ::1
first and the Podman containers bind IPv4 only.

Every statement here is either a literal string or uses psycopg2 parameter
binding. No value is ever interpolated into SQL text.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import psycopg2
import psycopg2.extras


def _setting(*names: str, default: str | None = None) -> str | None:
    """Return the first environment variable that is set.

    Args:
        *names: Environment variable names to try, in order.
        default: Value to return when none are set.

    Returns:
        The resolved value, or ``default``.
    """
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


def connection_kwargs() -> dict[str, Any]:
    """Build psycopg2 connection parameters from the environment.

    Returns:
        Keyword arguments for ``psycopg2.connect``.
    """
    kwargs: dict[str, Any] = {
        "host": _setting("PGHOST", "POSTGRES_HOST", default="127.0.0.1"),
        "port": int(_setting("PGPORT", "POSTGRES_PORT", default="5432")),
        "dbname": _setting("PGDATABASE", "POSTGRES_DB", default="boarddocs"),
        "user": _setting("PGUSER", "POSTGRES_USER", default="boarddocs"),
    }
    password = _setting("PGPASSWORD", "POSTGRES_PASSWORD")
    if password:
        kwargs["password"] = password
    return kwargs


@contextmanager
def connect(readonly: bool = False) -> Iterator[Any]:
    """Open a connection, committing on success and rolling back on error.

    Args:
        readonly: Open the session read-only. Use for every query path that
            touches ``documents``, ``chunks`` or anything outside ``facts`` --
            it makes the read-only contract enforced by Postgres rather than
            assumed by the caller.

    Yields:
        An open psycopg2 connection.
    """
    conn = psycopg2.connect(**connection_kwargs())
    try:
        if readonly:
            conn.set_session(readonly=True)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def query(sql: str, params: tuple | None = None) -> list[tuple]:
    """Run a read-only query against the corpus.

    Args:
        sql: SQL text with ``%s`` placeholders.
        params: Bound parameters.

    Returns:
        All result rows.
    """
    with connect(readonly=True) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()


def query_dicts(sql: str, params: tuple | None = None) -> list[dict]:
    """Run a read-only query returning dict rows.

    Args:
        sql: SQL text with ``%s`` placeholders.
        params: Bound parameters.

    Returns:
        All result rows as dicts.
    """
    with connect(readonly=True) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]
