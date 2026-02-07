
import sys
import os
from datetime import date as dt_date
from datetime import datetime
import logging

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.api.reports import trigger_daily_report 
from app.core.database import get_db

# Setup basic logging to see output
logging.basicConfig(level=logging.INFO)

print("Initializing DB session...")
db_gen = get_db()
db = next(db_gen)

print("Attempting to send daily report...")
try:
    result = trigger_daily_report(
        to_email="iamvimal3107@gmail.com", 
        target_date=dt_date.today(), 
        secret="INTELLICORE-SECRET", 
        db=db
    )
    print("Success!", result)
except Exception as e:
    print(f"Failed to send email: {e}")
finally:
    db.close()
