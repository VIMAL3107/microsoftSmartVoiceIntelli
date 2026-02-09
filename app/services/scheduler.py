
import time
import logging
import threading
from datetime import datetime
from app.core.config import REPORT_SCHEDULE_ENABLED, REPORT_INTERVAL_MINUTES, REPORT_EMAIL_TO, REPORT_DAILY_TIME
from app.core.database import get_db
from app.api.reports import trigger_batch_report 

logger = logging.getLogger(__name__)

def send_report(to_email):
    """Refactored sending logic to be reusable"""
    logger.info(f"SCHEDULER: Triggering Batch Report to {to_email}")
    db_gen = get_db()
    db = next(db_gen)
    try:
        trigger_batch_report(
            to_email=to_email,
            secret="INTELLICORE-SECRET",
            db=db
        )
    except Exception as exc:
        logger.error(f"SCHEDULER ERROR: {exc}")
    finally:
        db.close()

def run_scheduler_loop():
    """Background loop to check schedule and send emails."""
    time.sleep(10)  # Initial delay
    logger.info("SCHEDULER: Started.")

    last_run_date = None

    while True:
        try:
            # Config reloading removed as we are moving away from config.enc
            t_cfg = {}

            # Fallback to Environ Variables/Global Config
            s_enabled = t_cfg.get("enabled", REPORT_SCHEDULE_ENABLED)
            interval = float(t_cfg.get("interval_minutes", REPORT_INTERVAL_MINUTES))
            daily_time = t_cfg.get("daily_time", REPORT_DAILY_TIME) # Format "HH:MM" e.g. "20:00"
            to_email = t_cfg.get("email_to", REPORT_EMAIL_TO)

            if s_enabled:

                if not to_email:
                    time.sleep(60)
                    continue

                # MODE 1: Specific Daily Time (e.g. "20:00")
                if daily_time:
                     now = datetime.now()
                     current_time_str = now.strftime("%H:%M")
                     today_date = now.date()

                     # logger.debug(f"Checking time: {current_time_str} vs target {daily_time}")

                     if current_time_str == daily_time:
                         if last_run_date != today_date:
                             logger.info(f"SCHEDULER: Specific Time {daily_time} reached. Sending report.")
                             send_report(to_email)
                             last_run_date = today_date
                             
                             # Wait a bit to ensure we don't double trigger in the same minute 
                             # (though last_run_date check handles it, sleeping helps CPU)
                             time.sleep(60) 
                     
                     # Check every 30s to be accurate within the minute
                     time.sleep(30)
                
                # MODE 2: Interval (Fallback if no daily_time set)
                elif interval > 0:
                    logger.info(f"SCHEDULER: Sleeping for {interval} minutes...")
                    time.sleep(interval * 60)
                    send_report(to_email)
                
                else:
                    time.sleep(60)

            else:
                time.sleep(60)

        except Exception as e:
            logger.error(f"SCHEDULER CRASH: {e}")
            time.sleep(60)

def start_scheduler():
    t = threading.Thread(target=run_scheduler_loop, daemon=True)
    t.start()
