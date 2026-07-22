from __future__ import annotations

import sqlite3

SCHEMA_VERSION = 1


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
    ]
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
