import sqlite3
import os
import glob
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

DB_PATH = "voice_translate.db"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Assuming script is in scripts/
if not os.path.exists(os.path.join(PROJECT_ROOT, DB_PATH)):
    # Fallback if running from root
    PROJECT_ROOT = os.getcwd()

def clean_db():
    db_file = os.path.join(PROJECT_ROOT, DB_PATH)
    if not os.path.exists(db_file):
        logger.warning(f"Database file not found at {db_file}")
        return

    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        # Count before delete
        cursor.execute("SELECT COUNT(*) FROM call_analytics")
        count = cursor.fetchone()[0]
        
        # Delete
        cursor.execute("DELETE FROM call_analytics")
        conn.commit()
        
        logger.info(f"Database cleaned. Removed {count} records from 'call_analytics'.")
        conn.close()
    except Exception as e:
        logger.error(f"Error cleaning database: {e}")

def clean_excel():
    patterns = ["Batch_Report_*.xlsx", "Daily_Report_*.xlsx", "Call_Analytics.xlsx"]
    count = 0
    for pattern in patterns:
        files = glob.glob(os.path.join(PROJECT_ROOT, pattern))
        for f in files:
            try:
                os.remove(f)
                logger.info(f"Deleted file: {os.path.basename(f)}")
                count += 1
            except Exception as e:
                logger.error(f"Failed to delete {f}: {e}")
    
    if count == 0:
        logger.info("No Excel report files found to delete.")
    else:
        logger.info(f"Deleted {count} Excel files.")

if __name__ == "__main__":
    print(f"Cleaning system in {PROJECT_ROOT}...")
    clean_db()
    clean_excel()
    print("Cleanup complete.")
