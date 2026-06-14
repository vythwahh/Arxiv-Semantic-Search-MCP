CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    email TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS search_sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    query TEXT NOT NULL,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP,
    dwell_seconds REAL DEFAULT 0,
    scroll_seconds REAL DEFAULT 0,
    result_count INTEGER DEFAULT 0,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE IF NOT EXISTS paper_interactions (
    interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    paper_id TEXT NOT NULL,
    paper_title TEXT,
    paper_abstract TEXT,
    paper_categories TEXT,
    clicked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    read_seconds REAL DEFAULT 0,
    FOREIGN KEY (session_id) REFERENCES search_sessions(session_id)
);

CREATE TABLE IF NOT EXISTS keyword_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    keyword TEXT NOT NULL,
    session_id TEXT NOT NULL,
    frequency INTEGER DEFAULT 1,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES search_sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_search_sessions_user ON search_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_paper_interactions_user ON paper_interactions(user_id);
CREATE INDEX IF NOT EXISTS idx_keyword_events_user ON keyword_events(user_id);
CREATE INDEX IF NOT EXISTS idx_keyword_events_keyword ON keyword_events(keyword);
