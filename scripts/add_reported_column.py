import sqlite3
import os

DB_PATH = "voice_translate.db"

def upgrade_db():
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found: {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # Check if column exists
        cursor.execute("PRAGMA table_info(call_analytics)")
        columns = [info[1] for info in cursor.fetchall()]
        
        if "is_reported" in columns:
            print("✅ Column 'is_reported' already exists.")
        else:
            print("⚠️ Adding 'is_reported' column...")
            cursor.execute("ALTER TABLE call_analytics ADD COLUMN is_reported BOOLEAN DEFAULT 0")
            print("✅ Column added successfully.")
            
        conn.commit()
    except Exception as e:
        print(f"❌ Error updating database: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    upgrade_db()
