from app.core.database import get_db, Base, engine
from app.models.analytics import CallAnalytics
from sqlalchemy import text

def clear_data():
    print("🧹 Cleaning database tables...")
    try:
        # Create session
        db_gen = get_db()
        db = next(db_gen)
        
        # Delete all rows
        db.execute(text("DELETE FROM emails"))
        db.execute(text("DELETE FROM call_analytics"))
        db.execute(text("DELETE FROM users"))
        db.commit()
        
        # Reset Auto Increment (Optional, specific to SQLite)
        db.execute(text("DELETE FROM sqlite_sequence WHERE name='emails'"))
        db.execute(text("DELETE FROM sqlite_sequence WHERE name='call_analytics'"))
        db.execute(text("DELETE FROM sqlite_sequence WHERE name='users'"))
        db.commit()
        
        print("✅ Database cleared successfully!")
    except Exception as e:
        print(f"❌ Error cleaning DB: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    clear_data()
