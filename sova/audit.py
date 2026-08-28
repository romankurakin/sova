"""Read-only integrity checks for a Sova project database."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass

from sova import config

_EXPECTED_TABLES = {
    "_sqliteai_vector",
    "documents",
    "sections",
    "chunks",
    "chunk_contexts",
    "query_cache",
    "index_meta",
    "chunks_fts",
    "chunks_fts_data",
    "chunks_fts_idx",
    "chunks_fts_docsize",
    "chunks_fts_config",
    "chunks_fts_content",
}
_EXPECTED_META_KEYS = {
    "pipeline.context.signature",
    "pipeline.embedding.signature",
    "pipeline.chunk.signature",
}
_REQUIRED_COLUMNS = {
    "documents": {"id", "name"},
    "sections": {"id"},
    "chunks": {
        "id",
        "doc_id",
        "section_id",
        "section_path",
        "search_text",
        "text",
        "embedding",
    },
    "chunk_contexts": {"chunk_id", "context", "model"},
    "query_cache": {
        "embedding",
        "vector_results",
        "created_at",
        "model",
        "candidate_count",
    },
    "index_meta": {"key", "value"},
    "chunks_fts": {"search_text"},
}


@dataclass(frozen=True)
class Finding:
    """One actionable database audit result."""

    code: str
    message: str
    count: int = 1


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}


def audit_database(
    conn: sqlite3.Connection,
    *,
    expected_signatures: Mapping[str, str],
    expected_schema_version: int,
) -> list[Finding]:
    """Inspect an already-open connection without issuing any writes."""
    findings: list[Finding] = []
    tables = _tables(conn)

    missing = sorted(
        table
        for table in (
            "documents",
            "sections",
            "chunks",
            "chunk_contexts",
            "query_cache",
            "index_meta",
            "chunks_fts",
        )
        if table not in tables
    )
    if missing:
        findings.append(
            Finding(
                "schema.missing_tables",
                f"Missing tables: {', '.join(missing)}",
                len(missing),
            )
        )
        return findings

    malformed = []
    for table, required_columns in _REQUIRED_COLUMNS.items():
        absent = sorted(required_columns - _columns(conn, table))
        if absent:
            malformed.append(f"{table} ({', '.join(absent)})")
    if malformed:
        findings.append(
            Finding(
                "schema.missing_columns",
                "Missing columns: " + "; ".join(malformed),
                len(malformed),
            )
        )
        return findings

    unknown = sorted(
        table
        for table in tables - _EXPECTED_TABLES
        if not table.startswith("sqlite_")
        and not table.startswith("vector_")
        and not table.startswith("vector0_")
    )
    if unknown:
        findings.append(
            Finding(
                "schema.unknown_tables",
                f"Unknown tables: {', '.join(unknown)}",
                len(unknown),
            )
        )

    schema_version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    if schema_version != expected_schema_version:
        findings.append(
            Finding(
                "schema.version",
                f"Schema version is {schema_version}; expected {expected_schema_version}",
            )
        )

    stored_signatures = {
        str(key): str(value)
        for key, value in conn.execute("SELECT key, value FROM index_meta").fetchall()
    }
    stale_signature_keys = [
        key
        for key, expected in expected_signatures.items()
        if stored_signatures.get(key) != expected
    ]
    if stale_signature_keys:
        findings.append(
            Finding(
                "metadata.pipeline_mismatch",
                "Pipeline metadata does not match this Sova build: "
                + ", ".join(sorted(stale_signature_keys)),
                len(stale_signature_keys),
            )
        )

    integrity = str(conn.execute("PRAGMA integrity_check").fetchone()[0])
    if integrity != "ok":
        findings.append(Finding("database.integrity", integrity))

    foreign_key_issues = conn.execute("PRAGMA foreign_key_check").fetchall()
    if foreign_key_issues:
        findings.append(
            Finding(
                "database.foreign_keys",
                "Foreign-key violations detected",
                len(foreign_key_issues),
            )
        )

    orphan_sections = conn.execute(
        """
        SELECT COUNT(*) FROM chunks c
        LEFT JOIN sections s ON s.id = c.section_id
        WHERE c.section_id IS NOT NULL AND s.id IS NULL
        """
    ).fetchone()[0]
    if orphan_sections:
        findings.append(
            Finding(
                "chunks.orphan_sections",
                "Chunks reference missing sections",
                int(orphan_sections),
            )
        )

    document_columns = _columns(conn, "documents")
    if "source_signature" not in document_columns:
        findings.append(
            Finding(
                "documents.missing_provenance",
                "Documents do not record their source signature",
            )
        )
    else:
        unsigned_documents = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE source_signature IS NULL"
        ).fetchone()[0]
        if unsigned_documents:
            findings.append(
                Finding(
                    "documents.unsigned",
                    "Documents have no source signature",
                    int(unsigned_documents),
                )
            )

    context_columns = _columns(conn, "chunk_contexts")
    if "pipeline_signature" not in context_columns:
        findings.append(
            Finding(
                "contexts.missing_provenance",
                "Context rows do not record their pipeline signature",
            )
        )
    else:
        context_sig = expected_signatures["pipeline.context.signature"]
        stale_contexts = conn.execute(
            """
            SELECT COUNT(*) FROM chunk_contexts
            WHERE TRIM(context) = '' OR COALESCE(pipeline_signature, '') <> ?
            """,
            (context_sig,),
        ).fetchone()[0]
        if stale_contexts:
            findings.append(
                Finding(
                    "contexts.stale",
                    "Context rows are empty or belong to another pipeline",
                    int(stale_contexts),
                )
            )

    old_models = conn.execute(
        "SELECT COUNT(*) FROM chunk_contexts WHERE model <> ?",
        (config.CONTEXT_MODEL,),
    ).fetchone()[0]
    if old_models:
        findings.append(
            Finding(
                "contexts.mixed_models",
                "Context rows use a different model",
                int(old_models),
            )
        )

    chunk_columns = _columns(conn, "chunks")
    if "embedding_signature" not in chunk_columns:
        findings.append(
            Finding(
                "embeddings.missing_provenance",
                "Embedding rows do not record their pipeline signature",
            )
        )
    else:
        embed_sig = expected_signatures["pipeline.embedding.signature"]
        stale_embeddings = conn.execute(
            """
            SELECT COUNT(*) FROM chunks
            WHERE embedding IS NOT NULL
              AND COALESCE(embedding_signature, '') <> ?
            """,
            (embed_sig,),
        ).fetchone()[0]
        if stale_embeddings:
            findings.append(
                Finding(
                    "embeddings.stale",
                    "Embeddings belong to another pipeline",
                    int(stale_embeddings),
                )
            )

    invalid_embeddings = conn.execute(
        """
        SELECT COUNT(*) FROM chunks
        WHERE embedding IS NOT NULL AND LENGTH(embedding) <> ?
        """,
        (config.EMBEDDING_DIM * 4,),
    ).fetchone()[0]
    if invalid_embeddings:
        findings.append(
            Finding(
                "embeddings.invalid_size",
                "Embedding blobs have an unexpected size",
                int(invalid_embeddings),
            )
        )

    if "chunks_fts" in tables:
        try:
            chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            fts = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
            if chunks != fts:
                findings.append(
                    Finding(
                        "fts.row_mismatch",
                        f"FTS has {fts} rows for {chunks} chunks",
                        abs(int(chunks) - int(fts)),
                    )
                )
            sample_mismatches = 0
            for chunk_id, text in conn.execute(
                "SELECT id, search_text FROM chunks ORDER BY id LIMIT 32"
            ).fetchall():
                terms = re.findall(r"[^\W_]{2,}", str(text), flags=re.UNICODE)
                if not terms:
                    continue
                term = terms[0].replace('"', '""')
                indexed = conn.execute(
                    """
                    SELECT 1 FROM chunks_fts
                    WHERE chunks_fts MATCH ? AND rowid = ?
                    LIMIT 1
                    """,
                    (f'"{term}"', chunk_id),
                ).fetchone()
                if indexed is None:
                    sample_mismatches += 1
            if sample_mismatches:
                findings.append(
                    Finding(
                        "fts.sample_mismatch",
                        "FTS index does not match sampled chunk content",
                        sample_mismatches,
                    )
                )
        except sqlite3.Error as exc:
            findings.append(
                Finding(
                    "fts.unavailable",
                    f"FTS index could not be queried: {exc}",
                )
            )

    unknown_meta = [
        str(row[0])
        for row in conn.execute("SELECT key FROM index_meta ORDER BY key").fetchall()
        if str(row[0]) not in _EXPECTED_META_KEYS
    ]
    if unknown_meta:
        findings.append(
            Finding(
                "metadata.unknown_keys",
                f"Unknown metadata: {', '.join(unknown_meta)}",
                len(unknown_meta),
            )
        )
    return findings
