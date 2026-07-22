from __future__ import annotations

import sqlite3

SCHEMA_VERSION = 2


MIGRATIONS: dict[int, list[str]] = {
    1: [
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS projects (
            project_id TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS works (
            work_id TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            doi TEXT,
            arxiv_id TEXT,
            normalized_title TEXT NOT NULL UNIQUE
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS paper_versions (
            version_id TEXT PRIMARY KEY,
            work_id TEXT NOT NULL,
            data TEXT NOT NULL,
            source_url TEXT,
            content_hash TEXT,
            FOREIGN KEY(work_id) REFERENCES works(work_id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS papers (
            work_id TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            title TEXT NOT NULL,
            abstract TEXT,
            year INTEGER,
            doi TEXT,
            arxiv_id TEXT,
            source TEXT,
            FOREIGN KEY(work_id) REFERENCES works(work_id)
        )
        """,
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
            work_id UNINDEXED,
            title,
            abstract,
            authors,
            tokenize = 'porter unicode61'
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS evidence (
            evidence_id TEXT PRIMARY KEY,
            work_id TEXT NOT NULL,
            version_id TEXT NOT NULL,
            data TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            FOREIGN KEY(work_id) REFERENCES works(work_id),
            FOREIGN KEY(version_id) REFERENCES paper_versions(version_id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS claims (
            claim_id TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            type TEXT NOT NULL,
            support_status TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS hypotheses (
            hypothesis_id TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            status TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            status TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS run_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            data TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            question TEXT NOT NULL,
            data TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """,
    ],
    2: [
        """
        CREATE TABLE IF NOT EXISTS work_identifiers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            work_id TEXT NOT NULL,
            namespace TEXT NOT NULL,
            identifier TEXT NOT NULL,
            normalized_identifier TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(namespace, normalized_identifier),
            FOREIGN KEY(work_id) REFERENCES works(work_id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS paper_aliases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            work_id TEXT NOT NULL,
            alias_type TEXT NOT NULL,
            value TEXT NOT NULL,
            normalized_value TEXT NOT NULL,
            source TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(work_id, alias_type, normalized_value),
            FOREIGN KEY(work_id) REFERENCES works(work_id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS source_hits (
            hit_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            query_id TEXT NOT NULL,
            source TEXT NOT NULL,
            source_id TEXT,
            work_id TEXT,
            data TEXT NOT NULL,
            retrieved_at TEXT NOT NULL,
            FOREIGN KEY(work_id) REFERENCES works(work_id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS retrieval_provenance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            work_id TEXT NOT NULL,
            source TEXT NOT NULL,
            source_id TEXT,
            query_id TEXT NOT NULL,
            rank INTEGER,
            raw_score REAL,
            weight REAL NOT NULL DEFAULT 1.0,
            retrieved_at TEXT NOT NULL,
            UNIQUE(run_id, work_id, source, source_id, query_id),
            FOREIGN KEY(work_id) REFERENCES works(work_id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS duplicate_candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            left_work_id TEXT NOT NULL,
            right_work_id TEXT NOT NULL,
            decision TEXT NOT NULL,
            reason TEXT NOT NULL,
            confidence REAL NOT NULL,
            signals TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(left_work_id, right_work_id, decision),
            FOREIGN KEY(left_work_id) REFERENCES works(work_id),
            FOREIGN KEY(right_work_id) REFERENCES works(work_id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS tool_executions (
            execution_id TEXT PRIMARY KEY,
            run_id TEXT,
            tool_name TEXT NOT NULL,
            status TEXT NOT NULL,
            started_at TEXT NOT NULL,
            completed_at TEXT,
            latency_ms REAL,
            attempts INTEGER NOT NULL DEFAULT 1,
            error_type TEXT,
            data TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS budget_usage (
            run_id TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS run_manifests (
            run_id TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """,
    ],
}


def migrate(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    applied = {
        row[0] for row in connection.execute("SELECT version FROM schema_migrations").fetchall()
    }
    for version in sorted(MIGRATIONS):
        if version in applied:
            continue
        for statement in MIGRATIONS[version]:
            connection.execute(statement)
        connection.execute("INSERT INTO schema_migrations(version) VALUES (?)", (version,))
    connection.commit()
