-- Model Versions Table
-- Tracks all trained model versions with their metadata

CREATE TABLE IF NOT EXISTS model_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL UNIQUE,
    gender TEXT NOT NULL CHECK(gender IN ('male', 'female')),
    format_type TEXT NOT NULL DEFAULT 'T20' CHECK(format_type IN ('T20', 'ODI')),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    model_path TEXT NOT NULL,
    normalizer_path TEXT NOT NULL,
    data_earliest_date TEXT,
    data_latest_date TEXT,
    training_samples INTEGER,
    training_duration_seconds INTEGER,
    model_size_mb REAL,
    accuracy_metrics TEXT,
    is_active BOOLEAN NOT NULL DEFAULT 0,
    notes TEXT
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_model_versions_gender ON model_versions(gender);
CREATE INDEX IF NOT EXISTS idx_model_versions_is_active ON model_versions(is_active);
CREATE INDEX IF NOT EXISTS idx_model_versions_created_at ON model_versions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_model_versions_format ON model_versions(format_type);

