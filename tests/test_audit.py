"""Read-only database audit tests."""

import sqlite3

from sova import config
from sova.audit import audit_database
from sova.db import SCHEMA_VERSION, _migrate_schema

_EXPECTED_SIGNATURES = {
    "pipeline.context.signature": "context-v1",
    "pipeline.embedding.signature": "embed-v1",
    "pipeline.chunk.signature": "chunk-v1",
}


def _audit(conn: sqlite3.Connection):
    return audit_database(
        conn,
        expected_signatures=_EXPECTED_SIGNATURES,
        expected_schema_version=SCHEMA_VERSION,
    )


def _database() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY, name TEXT NOT NULL, path TEXT NOT NULL,
            expected_chunks INTEGER, source_signature TEXT, chunk_signature TEXT
        );
        CREATE TABLE sections (
            id INTEGER PRIMARY KEY, doc_id INTEGER NOT NULL, title TEXT NOT NULL
        );
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY, doc_id INTEGER NOT NULL, section_id INTEGER,
            section_path TEXT NOT NULL, text TEXT NOT NULL,
            search_text TEXT NOT NULL, embedding BLOB, embedding_signature TEXT,
            FOREIGN KEY (doc_id) REFERENCES documents(id)
        );
        CREATE TABLE chunk_contexts (
            chunk_id INTEGER PRIMARY KEY, context TEXT NOT NULL, model TEXT NOT NULL,
            pipeline_signature TEXT NOT NULL,
            FOREIGN KEY (chunk_id) REFERENCES chunks(id)
        );
        CREATE TABLE query_cache (
            id INTEGER PRIMARY KEY, embedding BLOB, vector_results BLOB,
            created_at REAL, model TEXT, candidate_count INTEGER
        );
        CREATE TABLE index_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE VIRTUAL TABLE chunks_fts USING fts5(search_text);
        INSERT INTO index_meta VALUES
            ('pipeline.context.signature', 'context-v1'),
            ('pipeline.embedding.signature', 'embed-v1'),
            ('pipeline.chunk.signature', 'chunk-v1');
        INSERT INTO documents
        VALUES (1, 'doc', '/tmp/doc.md', 1, 'source-v1', 'chunk-v1');
        INSERT INTO sections VALUES (1, 1, 'Section');
        """
    )
    conn.execute(
        "INSERT INTO chunks VALUES (1, 1, 1, 'Section', 'text', 'Context. text', ?, 'embed-v1')",
        (bytes(config.EMBEDDING_DIM * 4),),
    )
    conn.execute(
        "INSERT INTO chunk_contexts VALUES (1, 'Context.', ?, 'context-v1')",
        (config.CONTEXT_MODEL,),
    )
    conn.execute(
        "INSERT INTO chunks_fts(rowid, search_text) VALUES (1, 'Context. text')"
    )
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.commit()
    return conn


def test_healthy_database_has_no_findings_and_audit_makes_no_writes():
    conn = _database()
    before = conn.total_changes
    write_actions = {
        sqlite3.SQLITE_INSERT,
        sqlite3.SQLITE_UPDATE,
        sqlite3.SQLITE_DELETE,
        sqlite3.SQLITE_CREATE_TABLE,
        sqlite3.SQLITE_DROP_TABLE,
        sqlite3.SQLITE_ALTER_TABLE,
    }

    def deny_writes(action, _arg1, _arg2, _db, _trigger):
        return sqlite3.SQLITE_DENY if action in write_actions else sqlite3.SQLITE_OK

    conn.set_authorizer(deny_writes)
    assert _audit(conn) == []
    assert conn.total_changes == before
    conn.close()


def test_audit_reports_unknown_and_stale_state():
    conn = _database()
    conn.execute("CREATE TABLE scratch_results (id INTEGER)")
    conn.execute("UPDATE chunk_contexts SET pipeline_signature = 'old', model = 'old'")
    conn.execute("DELETE FROM chunks_fts")
    conn.commit()

    codes = {finding.code for finding in _audit(conn)}

    assert "schema.unknown_tables" in codes
    assert "contexts.stale" in codes
    assert "contexts.mixed_models" in codes
    assert "fts.row_mismatch" in codes
    conn.close()


def test_audit_accepts_sqlite_vector_internal_tables():
    conn = _database()
    conn.execute("CREATE TABLE vector0_chunks_embedding (id INTEGER)")

    codes = {finding.code for finding in _audit(conn)}

    assert "schema.unknown_tables" not in codes
    conn.close()


def test_audit_treats_null_embedding_provenance_as_stale():
    conn = _database()
    conn.execute("UPDATE chunks SET embedding_signature = NULL")

    findings = _audit(conn)

    assert any(
        finding.code == "embeddings.stale" and finding.count == 1
        for finding in findings
    )
    conn.close()


def test_audit_rejects_uniformly_old_pipeline_state():
    conn = _database()
    conn.execute("UPDATE chunk_contexts SET pipeline_signature = 'old-context'")
    conn.execute("UPDATE chunks SET embedding_signature = 'old-embed'")
    conn.execute(
        "UPDATE index_meta SET value = 'old-' || value WHERE key LIKE 'pipeline.%'"
    )
    conn.commit()

    codes = {finding.code for finding in _audit(conn)}

    assert "metadata.pipeline_mismatch" in codes
    assert "contexts.stale" in codes
    assert "embeddings.stale" in codes
    conn.close()


def test_audit_requires_runtime_search_tables():
    conn = _database()
    conn.execute("DROP TABLE query_cache")

    findings = _audit(conn)

    assert findings[0].code == "schema.missing_tables"
    assert "query_cache" in findings[0].message
    conn.close()

    conn = _database()
    conn.execute("DROP TABLE chunks_fts")

    findings = _audit(conn)

    assert findings[0].code == "schema.missing_tables"
    assert "chunks_fts" in findings[0].message
    conn.close()


def test_schema_migration_adds_pipeline_provenance_idempotently():
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE documents (id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE chunks (id INTEGER PRIMARY KEY, embedding BLOB);
        CREATE TABLE chunk_contexts (
            chunk_id INTEGER PRIMARY KEY, context TEXT NOT NULL, model TEXT NOT NULL
        );
        """
    )

    _migrate_schema(conn)
    _migrate_schema(conn)

    chunk_columns = {row[1] for row in conn.execute("PRAGMA table_info(chunks)")}
    context_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(chunk_contexts)")
    }
    document_columns = {row[1] for row in conn.execute("PRAGMA table_info(documents)")}
    assert "embedding_signature" in chunk_columns
    assert "section_path" in chunk_columns
    assert "search_text" in chunk_columns
    assert "pipeline_signature" in context_columns
    assert "source_signature" in document_columns
    assert "chunk_signature" in document_columns
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    conn.close()


def test_schema_migration_rejects_a_database_from_a_newer_sova():
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        f"""
        CREATE TABLE documents (id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE chunks (id INTEGER PRIMARY KEY, embedding BLOB);
        CREATE TABLE chunk_contexts (
            chunk_id INTEGER PRIMARY KEY, context TEXT NOT NULL, model TEXT NOT NULL
        );
        PRAGMA user_version = {SCHEMA_VERSION + 1};
        """
    )

    try:
        _migrate_schema(conn)
    except RuntimeError as exc:
        assert "newer than supported" in str(exc)
    else:
        raise AssertionError("newer database schema was accepted")
    conn.close()
