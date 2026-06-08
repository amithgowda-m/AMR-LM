import sqlite3
import os
from datetime import datetime
import json

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'history.db')

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sequence TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            probabilities TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_prediction(sequence, prediction, confidence, probabilities):
    """Saves a prediction to the database and returns the new ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    # Save a preview of the sequence if it's very long
    seq_preview = sequence[:100] + "..." if len(sequence) > 100 else sequence
    
    cursor.execute('''
        INSERT INTO predictions (sequence, prediction, confidence, probabilities, timestamp)
        VALUES (?, ?, ?, ?, ?)
    ''', (seq_preview, prediction, confidence, json.dumps(probabilities), datetime.utcnow().isoformat() + "Z"))
    
    new_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return new_id

def get_history(limit=50):
    """Retrieves the most recent predictions from the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, sequence, prediction, confidence, probabilities, timestamp
        FROM predictions
        ORDER BY id DESC
        LIMIT ?
    ''', (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    
    history = []
    for row in rows:
        history.append({
            "id": row["id"],
            "sequence_preview": row["sequence"],
            "prediction": row["prediction"],
            "confidence": row["confidence"],
            "probabilities": json.loads(row["probabilities"]),
            "timestamp": row["timestamp"]
        })
    return history

# Initialize DB when the module is imported
init_db()
