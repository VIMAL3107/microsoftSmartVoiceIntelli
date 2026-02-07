import sqlite3
import os
from datetime import datetime

DB_PATH = "voice_translate.db"
XLSX_PATH = "Call_Analytics.xlsx"

def inspect():
    print(f"Inspecting Database: {DB_PATH}")
    if not os.path.exists(DB_PATH):
        print(f"❌ Database file not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # List tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [t[0] for t in cursor.fetchall()]
    print(f"✅ Tables found: {', '.join(tables)}")

    # Check CallAnalytics
    if 'call_analytics' in tables:
        print("\n📊 --- Table: call_analytics ---")
        count = cursor.execute("SELECT count(*) FROM call_analytics").fetchone()[0]
        print(f"Total Records: {count}")
        
        if count > 0:
            print("\nLast 3 records:")
            rows = cursor.execute("SELECT id, session_id, created_at, detected_language FROM call_analytics ORDER BY id DESC LIMIT 3").fetchall()
            print(f"{'ID':<5} | {'Created At':<20} | {'Lang':<5} | {'Session ID'}")
            print("-" * 60)
            for row in rows:
                print(f"{row['id']:<5} | {row['created_at']:<20} | {row['detected_language']:<5} | {row['session_id']}")
    else:
        print("\n❌ Table 'call_analytics' NOT found.")

    conn.close()

    # Check Excel File
    print(f"\nChecking Excel File: {XLSX_PATH}")
    if os.path.exists(XLSX_PATH):
        mod_time = datetime.fromtimestamp(os.path.getmtime(XLSX_PATH))
        print(f"✅ File exists. Last modified: {mod_time}")
    else:
        print("❌ File does not exist yet.")

if __name__ == "__main__":
    inspect()
