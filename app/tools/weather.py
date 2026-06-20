from langchain_core.tools import tool
import httpx
import json
import urllib.parse

def wmo_to_condition(code: int) -> str:
    if code == 0:
        return "晴"
    elif code <= 3:
        return "多云"
    elif code <= 49:
        return "雾"
    elif code <= 59:
        return "毛毛雨"
    elif code <= 69:
        return "雨"
    elif code <= 79:
        return "雪"
    elif code <= 82:
        return "阵雨"
    elif code <= 86:
        return "阵雪"
    elif code <= 99:
        return "雷暴"
    else:
        return "未知"

@tool
def weather(location: str = None, latitude: float = None, longitude: float = None) -> dict:
    """Get weather information for a location. Use this when the user asks about the weather.
    When user mentions a city name (e.g. "北京", "Shanghai"), use the location field.
    
    Args:
        location: City name, e.g. "北京", "Shanghai"
        latitude: Latitude coordinate (optional)
        longitude: Longitude coordinate (optional)
    """
    lat, lon = latitude, longitude
    location_name = ""

    # 1. Resolve location name via Geocoding if provided
    if location:
        try:
            geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.parse.quote(location)}&count=1&language=zh"
            r = httpx.get(geo_url, timeout=10.0)
            if r.status_code == 200:
                geo_data = r.json()
                if geo_data.get("results"):
                    result = geo_data["results"][0]
                    lat = result["latitude"]
                    lon = result["longitude"]
                    location_name = result["name"] + ", " + result.get("country", "")
                else:
                    return {"error": f"Location '{location}' not found"}
            else:
                return {"error": f"Geocoding API error: {r.status_code}"}
        except Exception as e:
            return {"error": f"Geocoding failed: {str(e)}"}
    elif lat is not None and lon is not None:
        location_name = f"{lat:.4f}, {lon:.4f}"
    else:
        return {"error": "No location or coordinates provided"}

    # 2. Query weather forecast
    try:
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,cloud_cover,wind_speed_10m,wind_direction_10m,weather_code&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code,wind_speed_10m_max&forecast_days=7"
        r = httpx.get(weather_url, timeout=10.0)
        if r.status_code != 200:
            return {"error": f"Weather API error: {r.status_code}"}
        
        data = r.json()
        current = data["current"]
        daily = data["daily"]
        
        current_condition = wmo_to_condition(current["weather_code"])
        
        forecast = []
        for i, time_val in enumerate(daily["time"]):
            day_condition = wmo_to_condition(daily["weather_code"][i])
            day_data = {
                "date": time_val,
                "high": daily["temperature_2m_max"][i],
                "low": daily["temperature_2m_min"][i],
                "condition": day_condition,
                "precipitation": daily["precipitation_sum"][i],
                "wind_speed_max": daily["wind_speed_10m_max"][i]
            }
            forecast.append(day_data)
            
        weather_data_struct = {
            "location": location_name,
            "current": {
                "temp": current["temperature_2m"],
                "feels_like": current["apparent_temperature"],
                "humidity": current["relative_humidity_2m"],
                "condition": current_condition,
                "wind_speed": current["wind_speed_10m"],
                "wind_direction": current["wind_direction_10m"],
                "precipitation": current["precipitation"],
                "cloud_cover": current["cloud_cover"]
            },
            "forecast": forecast
        }
        
        weather_json_str = json.dumps(weather_data_struct, ensure_ascii=False)
        summary = f"Current weather in {location_name}: {current_condition}, {current['temperature_2m']}°C (feels like {current['apparent_temperature']}°C), humidity {current['relative_humidity_2m']}%, wind {current['wind_speed_10m']} km/h. 7-day forecast included."
        
        return {
            "weather_data": weather_json_str,
            "summary": summary
        }
    except Exception as e:
        return {"error": f"Weather fetch failed: {str(e)}"}
