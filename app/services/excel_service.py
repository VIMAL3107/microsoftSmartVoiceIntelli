
import pandas as pd
from sqlalchemy import text
import logging
import os
import json

# Configure logger
logger = logging.getLogger(__name__)

def export_call_analytics_to_excel(db_session, output_file="Call_Analytics.xlsx", target_date=None):
    """
    Exports data from the 'call_analytics' table to an Excel file.
    
    Args:
        db_session: The SQLAlchemy database session.
        output_file (str): The name of the output Excel file.
        target_date (date, optional): If provided, exports only records created on this date.
    """
    try:
        # Define the query
        if target_date:
            # Filter by date string matching (robust for SQLite text dates)
            # created_at format is usually YYYY-MM-DD HH:MM:SS.ssssss
            date_str = str(target_date)
            query = text("SELECT * FROM call_analytics WHERE created_at LIKE :date_pattern")
            params = {"date_pattern": f"{date_str}%"}
            df = pd.read_sql(query, db_session.bind, params=params)
        else:
            query = text("SELECT * FROM call_analytics")
            df = pd.read_sql(query, db_session.bind)
        
        # Check if DataFrame is empty
        if df.empty:
            logger.warning("No data found in 'call_analytics' table to export.")
            # Create an empty excel file or just return
            # Better to create empty so email attachement doesn't fail
            df = pd.DataFrame()

        # Parse JSON columns to ensure proper formatting
        json_cols = ['segments', 'agent_improvement_areas']
        for col in json_cols:
            if col in df.columns:
                def clean_json(x):
                    try:
                        obj = json.loads(x) if isinstance(x, str) else x
                        return json.dumps(obj, ensure_ascii=False, indent=2)
                    except Exception:
                        return x
                
                df[col] = df[col].apply(clean_json)
        
        # Export to Excel
        df.to_excel(output_file, index=False, engine='openpyxl')
        logger.info(f"Successfully exported call analytics to {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"Failed to export call analytics to Excel: {e}")
        raise

def export_and_mark_unreported_analytics(db_session, output_file="Batch_Report.xlsx"):
    """
    Exports UNREPORTED data from 'call_analytics' to Excel and marks them as reported.
    
    Args:
        db_session: The SQLAlchemy database session.
        output_file (str): The name of the output Excel file.
    
    Returns:
        str: The filename if successful and data found, None if no data.
    """
    try:
        # 1. Fetch unreported records
        # we check for 0 (False) or NULL to be safe
        query = text("SELECT * FROM call_analytics WHERE is_reported = 0 OR is_reported IS NULL")
        df = pd.read_sql(query, db_session.bind)
        
        # If no new data, return None
        if df.empty:
            logger.warning("No unreported data found. Skipping export.")
            return None
            
        # 2. Parse JSON columns (reuse logic)
        json_cols = ['segments', 'agent_improvement_areas']
        for col in json_cols:
            if col in df.columns:
                def clean_json(x):
                    try:
                        obj = json.loads(x) if isinstance(x, str) else x
                        return json.dumps(obj, ensure_ascii=False, indent=2)
                    except Exception:
                        return x
                
                df[col] = df[col].apply(clean_json)
        
        # 3. Export to Excel
        df.to_excel(output_file, index=False, engine='openpyxl')
        logger.info(f"Successfully exported {len(df)} unreported records to {output_file}")
        
        # 4. Mark these records as reported
        # Get IDs from the dataframe
        ids = df['id'].tolist()
        if ids:
            # Use ORM update for safety and database agnosticism
            from app.models.analytics import CallAnalytics
            db_session.query(CallAnalytics).filter(CallAnalytics.id.in_(ids)).update(
                {"is_reported": True}, 
                synchronize_session=False
            )
            db_session.commit()
            logger.info(f"Marked {len(ids)} records as reported.")
            
        return output_file
        
    except Exception as e:
        logger.error(f"Failed to export batch analytics: {e}")
        db_session.rollback()
        raise
