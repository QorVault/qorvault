# Security Policy

## Supported Versions

QorVault is distributed under BSL 1.1 (converting to AGPL-3.0-or-later
after four years per version).  Security updates are provided for the most recent
tagged release on the `main` branch.  Older releases are not supported
once a newer tag is published.

| Version | Supported          |
| ------- | ------------------ |
| latest  | :white_check_mark: |
| < latest | :x:               |

## Reporting a Vulnerability

If you discover a security vulnerability in QorVault, please report it
privately by emailing **donald@qorvault.com**.  Do not open a public
GitHub issue, pull request, or discussion for security reports — public
disclosure before a fix is available puts other deployers at risk.

Please include in your report:

- A description of the vulnerability and its impact
- Steps to reproduce, or a proof-of-concept if available
- The affected version (commit hash or tag)
- Your name and contact information for follow-up (optional but
  appreciated)

You will receive an acknowledgement within 3 business days.  We aim to
provide an initial assessment and remediation timeline within 7 business
days, and to ship a fix within 30 days for HIGH/CRITICAL findings or 90
days for MEDIUM findings.  Coordinated disclosure timelines beyond 90
days will be discussed case-by-case.

## Known Limitations

The following are documented design limitations rather than
vulnerabilities, but operators should understand them before deploying:

- The `enable_routing` flag on `POST /api/v1/query` opts in to an
  LLM-generated SQL execution path (`rag_api/rag_api/database_handler.py`).
  The default is `False` and the shipped landing-page UI does not enable
  it.  If you set it to `True`, the SQL validator is a deny-list rather
  than an AST allow-list — do not enable this path on an
  internet-exposed deployment without an authentication boundary in
  front of it (reverse proxy + auth, IP allow-list, or equivalent), and
  configure the database connection role with `SELECT`-only privileges
  scoped to the application tables.

- The RAG API has no built-in authentication.  Place it behind a
  reverse proxy or auth gateway when exposing it to networks outside
  your trust boundary.
