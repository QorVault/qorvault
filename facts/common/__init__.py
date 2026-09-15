"""Code shared by every package in the fact layer.

Anything in here is used by more than one fact package and must not depend
on any of them. Today that is path resolution: ``documents.file_path`` is
stale under more than one root, and every package that re-reads a source
PDF needs the same answer to "where does this file actually live".
"""
