from langchain_core.tools import tool
import datetime
import pytz

@tool
def get_time(timezone: str = "Asia/Shanghai") -> dict:
    """Get the current date and time, or convert between timezones. Use this when the user asks 
    about the current time, date, or time in a specific timezone.
    
    Args:
        timezone: Optional timezone, e.g. Asia/Shanghai, America/New_York. Defaults to Asia/Shanghai.
    """
    try:
        tz = pytz.timezone(timezone)
    except Exception:
        # Fallback to CST (UTC+8) if invalid timezone
        tz = pytz.timezone("Asia/Shanghai")
        
    now = datetime.datetime.now(tz)
    return {
        "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "weekday": now.strftime("%A"),
        "timezone": str(tz),
        "timestamp": int(now.timestamp())
    }
