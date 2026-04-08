-- BoardDocs RAG Infrastructure — Database Schema
-- This script is idempotent: safe to run multiple times.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS tenants (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id   VARCHAR(64) UNIQUE NOT NULL,
    name        VARCHAR(255) NOT NULL,
    base_url    TEXT NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS documents (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id           VARCHAR(64) NOT NULL REFERENCES tenants(tenant_id),
    external_id         VARCHAR(255),
    document_type       VARCHAR(64) NOT NULL,
    title               TEXT,
    content_raw         TEXT,
    content_text        TEXT,
    source_url          TEXT,
    file_path           TEXT,
    meeting_date        DATE,
    committee_name      VARCHAR(255),
    meeting_id          VARCHAR(255),
    agenda_item_id      VARCHAR(255),
    processing_status   VARCHAR(32) DEFAULT 'pending',
    processing_error    TEXT,
    ocr_applied         BOOLEAN DEFAULT FALSE,
    ocr_method          VARCHAR(100),
    ocr_confidence      FLOAT,
    page_count          INT,
    pages_ocr_count     INT,
    has_tables          BOOLEAN DEFAULT FALSE,
    table_count         INT DEFAULT 0,
    metadata            JSONB DEFAULT '{}',
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(tenant_id, external_id)
);

CREATE TABLE IF NOT EXISTS document_pages (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id         UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number         INT NOT NULL,
    extraction_method   VARCHAR(50) NOT NULL,
    content_text        TEXT,
    ocr_confidence      FLOAT,
    has_tables          BOOLEAN DEFAULT FALSE,
    table_data          JSONB,
    processing_time_ms  INT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(document_id, page_number)
);

CREATE TABLE IF NOT EXISTS chunks (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id           VARCHAR(64) NOT NULL REFERENCES tenants(tenant_id),
    document_id         UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index         INT NOT NULL,
    content             TEXT NOT NULL,
    token_count         INT,
    source_page         INT,
    contains_table      BOOLEAN DEFAULT FALSE,
    embedding_status    VARCHAR(32) DEFAULT 'pending',
    embedding_model     VARCHAR(128),
    qdrant_point_id     UUID,
    metadata            JSONB DEFAULT '{}',
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_documents_tenant       ON documents(tenant_id);
CREATE INDEX IF NOT EXISTS idx_documents_type         ON documents(document_type);
CREATE INDEX IF NOT EXISTS idx_documents_status       ON documents(processing_status);
CREATE INDEX IF NOT EXISTS idx_documents_date         ON documents(meeting_date);
CREATE INDEX IF NOT EXISTS idx_document_pages_doc     ON document_pages(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_tenant          ON chunks(tenant_id);
CREATE INDEX IF NOT EXISTS idx_chunks_document        ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_embed_status    ON chunks(embedding_status);

INSERT INTO tenants (tenant_id, name, base_url)
VALUES ('kent_sd', 'Kent School District', 'https://go.boarddocs.com/wa/ksdwa/Board.nsf/Public')
ON CONFLICT (tenant_id) DO NOTHING;

-- AI activity monitoring log — aggregates events from filesystem watcher,
-- auditd, terminal session logs, and git snapshots.
CREATE TABLE IF NOT EXISTS ai_activity_log (
    id SERIAL PRIMARY KEY,
    session_id UUID NOT NULL,
    session_start TIMESTAMPTZ NOT NULL,
    session_end TIMESTAMPTZ,
    ai_model VARCHAR(100),
    ai_version VARCHAR(50),
    action_sequence INTEGER,
    action_type VARCHAR(50) NOT NULL,
    action_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    file_path TEXT,
    command_executed TEXT,
    reasoning TEXT,
    outcome TEXT,
    working_directory TEXT,
    git_commit_hash VARCHAR(40),
    success BOOLEAN,
    error_message TEXT,
    duration_ms INTEGER,
    initiated_by VARCHAR(100) DEFAULT 'claude-code',
    project_name VARCHAR(100) DEFAULT 'ksd-boarddocs-rag',
    source VARCHAR(50),
    raw_event JSONB
);

CREATE INDEX IF NOT EXISTS idx_ai_activity_session
    ON ai_activity_log(session_id);
CREATE INDEX IF NOT EXISTS idx_ai_activity_timestamp
    ON ai_activity_log(action_timestamp);
CREATE INDEX IF NOT EXISTS idx_ai_activity_type
    ON ai_activity_log(action_type);
CREATE INDEX IF NOT EXISTS idx_ai_activity_file
    ON ai_activity_log(file_path);
