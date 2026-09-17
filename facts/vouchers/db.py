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

There is a second, **read-only** transport for sessions that have no
credential. See :func:`_psql_query_dicts`. It is opt-in, it can only read,
and it changes nothing about the default path.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from decimal import Decimal
from typing import Any

import psycopg2
import psycopg2.extras
from psycopg2.extensions import adapt


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


# ----------------------------------- credential-free read-only transport --
#
# WHY THIS EXISTS. The host reaches this database through a rootless Podman
# port-forward. Postgres therefore sees the connection arriving from the
# forwarder's address rather than from 127.0.0.1, so ``pg_hba.conf`` falls
# past its three ``trust`` lines to ``host all all all scram-sha-256`` and
# psycopg2 must present a password. Inside the container the first line,
# ``local all all trust``, applies and a query needs no credential at all.
#
# A session with no password can therefore still READ, and that matters for
# one specific thing: ``test_no_leaks.py`` is the acceptance test for the
# payee privacy control and it needs the corpus's withheld list. Without
# this it skips, and a check that did not run is not a check that succeeded.
#
# WHAT KEEPS IT HONEST:
#
# * **Opt-in.** Nothing changes unless ``VOUCHERS_DB_TRANSPORT=podman`` is
#   set. The default path is byte-for-byte what it was.
# * **Read-only by the server, not by convention.** Every statement is
#   wrapped in ``BEGIN; SET TRANSACTION READ ONLY;``, so Postgres refuses a
#   write on this transport even if one is passed in.
# * **No interpolation.** Parameters are rendered by psycopg2's own
#   adapters -- the same code that binds them on the psycopg2 path -- and
#   only for values where a connection-less adapter is provably exact
#   (see :func:`_quote`). Anything outside that raises.
# * **Order is pinned.** ``json_agg`` has no ordering guarantee over a
#   subquery, so the rows are numbered as they leave the inner query and
#   the aggregate is ordered on that number. Without this, ``ORDER BY``
#   in a caller's SQL would be advisory.

TRANSPORT_ENV = "VOUCHERS_DB_TRANSPORT"
CONTAINER_ENV = "VOUCHERS_DB_CONTAINER"
DEFAULT_CONTAINER = "boarddocs-postgres"

# Wraps a caller's SELECT so the whole result comes back as one JSON
# document. ``{sql}`` is the caller's own literal SQL text; no value is ever
# formatted into it.
_JSON_WRAPPER = (
    "SELECT coalesce(json_agg(row_to_json(_q) ORDER BY _q._rn), '[]'::json)\n"
    "FROM (SELECT *, row_number() OVER () AS _rn FROM (\n{sql}\n) _s) _q;"
)

_ROW_NUMBER_KEY = "_rn"

# Cached answer to "does this server treat a backslash literally?". One
# query per process; None until asked.
_SCS: bool | None = None


def use_psql_transport() -> bool:
    """Whether read queries should go through ``podman exec ... psql``.

    Returns:
        True when the operator has opted in explicitly.
    """
    return os.environ.get(TRANSPORT_ENV, "").strip().lower() == "podman"


def _standard_conforming_strings() -> bool:
    """Whether the server treats a backslash in a literal as an ordinary character.

    Asked once per process, with a query that takes no parameters, so there
    is no chicken-and-egg problem with the renderer below.

    Returns:
        True when ``standard_conforming_strings`` is on.
    """
    global _SCS
    if _SCS is None:
        rows = _psql_rows("SELECT current_setting('standard_conforming_strings') AS v")
        _SCS = bool(rows) and rows[0].get("v") == "on"
    return _SCS


def _quote_text(value: str) -> str:
    r"""Render one text parameter as a SQL literal.

    Under ``standard_conforming_strings = on`` -- the server default since
    PostgreSQL 9.1, and checked here rather than assumed -- the complete
    rule for a single-quoted literal is that ``'`` is written twice and
    every other character, backslash included, stands for itself.

    psycopg2's adapter is used as a cross-check wherever it is known to be
    exact without a connection, which is every string with no backslash. It
    cannot be used for the backslash case: with no connection it assumes
    ``standard_conforming_strings`` is OFF and doubles them, which would
    turn a regex parameter such as ``\\s+`` into something that matches
    different rows. That is the bug this function exists to avoid, and it
    was caught by the guard that used to sit here refusing the value.

    Args:
        value: The text to render.

    Returns:
        The SQL literal.

    Raises:
        ValueError: When the server is not in standard-conforming mode, or
            when the cross-check disagrees.
    """
    if not _standard_conforming_strings():
        raise ValueError(
            f"{TRANSPORT_ENV}=podman requires standard_conforming_strings=on; "
            f"this server has it off. Run with a database credential instead."
        )
    literal = "'" + value.replace("'", "''") + "'"
    if "\\" not in value and value.isascii():
        expected = adapt(value).getquoted().decode("ascii")
        if expected != literal:
            raise ValueError(f"literal rendering disagrees with psycopg2 for {value!r}")
    return literal


def _quote(value: Any) -> str:
    """Render one query parameter as a SQL literal.

    Text goes through :func:`_quote_text`. Everything else goes through
    psycopg2's own adapter, which needs no connection for numbers, booleans,
    NULL and dates.

    Args:
        value: A parameter value.

    Returns:
        The SQL literal for that value.

    Raises:
        ValueError: When the value is of a type this transport does not
            render.
    """
    if isinstance(value, list | tuple):
        return "ARRAY[" + ",".join(_quote(item) for item in value) + "]"
    if isinstance(value, str):
        return _quote_text(value)
    _check_renderable(value)
    return adapt(value).getquoted().decode("ascii")


def _check_renderable(value: Any) -> None:
    """Reject non-text values this transport does not render.

    Args:
        value: A scalar parameter value.

    Raises:
        ValueError: When the type is not one psycopg2 can adapt without a
            connection.
    """
    if value is None or isinstance(value, bool | int | Decimal):
        return
    if hasattr(value, "isoformat"):  # date / datetime
        return
    raise ValueError(f"{TRANSPORT_ENV}=podman does not render {type(value).__name__} parameters")


def _bind(sql: str, params: tuple | None) -> str:
    """Substitute rendered literals for the ``%s`` placeholders.

    Split-and-rejoin rather than ``%``-formatting, so a literal percent in
    a caller's SQL cannot be mistaken for a placeholder.

    Args:
        sql: SQL text with ``%s`` placeholders.
        params: Bound parameters.

    Returns:
        SQL text with every placeholder replaced by a rendered literal.

    Raises:
        ValueError: When the placeholder count and the parameter count
            disagree.
    """
    parts = sql.split("%s")
    supplied = tuple(params or ())
    if len(parts) - 1 != len(supplied):
        raise ValueError(f"expected {len(parts) - 1} parameters, got {len(supplied)}")
    out = [parts[0]]
    for literal, tail in zip((_quote(p) for p in supplied), parts[1:], strict=True):
        out.append(literal)
        out.append(tail)
    return "".join(out)


def _psql_query_dicts(sql: str, params: tuple | None = None) -> list[dict]:
    """Run one read-only query inside the database container.

    Args:
        sql: SQL text with ``%s`` placeholders.
        params: Bound parameters.

    Returns:
        All result rows as dicts, in the order the query produced them.

    Raises:
        RuntimeError: When podman is absent or psql exits non-zero.
    """
    return _psql_rows(_bind(sql, params))


def _psql_rows(sql_text: str) -> list[dict]:
    """Run already-rendered SQL inside the container and return its rows.

    Split out from :func:`_psql_query_dicts` so the one query that has to
    run BEFORE the parameter renderer is ready -- the
    ``standard_conforming_strings`` check -- can reach the server without
    going back through the renderer.

    Args:
        sql_text: Complete SQL with every literal already rendered.

    Returns:
        All result rows as dicts, in the order the query produced them.

    Raises:
        RuntimeError: When podman is absent or psql exits non-zero.
    """
    # Resolved to an absolute path rather than left as "podman", so what
    # runs cannot depend on PATH ordering at call time.
    podman = shutil.which("podman")
    if podman is None:
        raise RuntimeError(f"{TRANSPORT_ENV}=podman is set but podman is not on PATH")
    container = os.environ.get(CONTAINER_ENV) or DEFAULT_CONTAINER
    script = "BEGIN;\nSET TRANSACTION READ ONLY;\n" + _JSON_WRAPPER.format(sql=sql_text) + "\nCOMMIT;\n"
    # UTF-8 is pinned on both sides rather than inherited from the locale,
    # so a payee name outside ASCII travels unchanged.
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell, SQL on stdin
        [
            podman,
            "exec",
            "-i",
            "--env",
            "PGCLIENTENCODING=UTF8",
            container,
            "psql",
            "-U",
            _setting("PGUSER", "POSTGRES_USER", default="boarddocs"),
            "-d",
            _setting("PGDATABASE", "POSTGRES_DB", default="boarddocs"),
            "-At",
            "-q",
            "-v",
            "ON_ERROR_STOP=1",
        ],
        input=script.encode("utf-8"),
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"psql failed in container {container}: {result.stderr.decode('utf-8', 'replace').strip()[:500]}"
        )
    # parse_float=Decimal: row_to_json prints a numeric as a bare JSON
    # number, and reading it as a float would silently round money.
    rows = json.loads(result.stdout.decode("utf-8") or "[]", parse_float=Decimal)
    for row in rows:
        row.pop(_ROW_NUMBER_KEY, None)
    return rows


def query(sql: str, params: tuple | None = None) -> list[tuple]:
    """Run a read-only query against the corpus.

    Args:
        sql: SQL text with ``%s`` placeholders.
        params: Bound parameters.

    Returns:
        All result rows.
    """
    if use_psql_transport():
        # Tuples in the column order the caller asked for. dict preserves
        # insertion order and row_to_json emits the select list in order,
        # so this is the same ordering psycopg2 would have given.
        return [tuple(row.values()) for row in _psql_query_dicts(sql, params)]
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
    if use_psql_transport():
        return _psql_query_dicts(sql, params)
    with connect(readonly=True) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]
