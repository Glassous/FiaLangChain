from langchain_core.tools import tool
import httpx
from app.core.config import settings

from langchain_core.runnables import RunnableConfig

@tool
def web_search(query: str, source: str = "bocha", config: RunnableConfig = None) -> dict:
    """Search the web for information. Use this when you need current information, news, facts, 
    or details not in your training data.
    
    Args:
        query: The search query keywords
        source: Optional data source. 'bocha' for Chinese/domestic information, 'tavily' for international/English information. Defaults to 'bocha'.
    """
    configurable = config.get("configurable", {}) if config else {}
    tavily_key = configurable.get("tavily_api_key") or settings.TAVILY_API_KEY
    bocha_key = configurable.get("bocha_api_key") or settings.BOCHA_API_KEY

    if source == "tavily" and tavily_key:
        try:
            url = "https://api.tavily.com/search"
            payload = {
                "api_key": tavily_key,
                "query": query,
                "max_results": 5
            }
            r = httpx.post(url, json=payload, timeout=10.0)
            if r.status_code == 200:
                data = r.json()
                results = []
                for res in data.get("results", []):
                    results.append({
                        "title": res.get("title", ""),
                        "url": res.get("url", ""),
                        "snippet": res.get("content", "")
                    })
                return {"query": query, "results": results, "source": "tavily"}
            else:
                # Fallback to bocha if tavily fails or keys aren't set
                pass
        except Exception as e:
            pass

    # Default to Bocha Search
    if not bocha_key:
        return {"query": query, "results": "Search unavailable: API key not configured"}
        
    try:
        url = "https://api.bochaai.com/v1/web-search"
        headers = {
            "Authorization": f"Bearer {bocha_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "query": query,
            "count": 8
        }
        r = httpx.post(url, json=payload, headers=headers, timeout=10.0)
        if r.status_code != 200:
            return {"query": query, "results": f"Bocha AI search error: status {r.status_code}"}
            
        data = r.json()
        raw_results = data.get("data", {}).get("webPages", {}).get("value", [])
        results = []
        for item in raw_results:
            results.append({
                "title": item.get("name", ""),
                "url": item.get("url", ""),
                "snippet": item.get("snippet", ""),
                "site_name": item.get("siteName", ""),
                "site_icon": item.get("siteIcon", ""),
                "date_published": item.get("datePublished", "")
            })
        return {"query": query, "results": results, "source": "bocha"}
    except Exception as e:
        return {"query": query, "results": f"Search failed: {str(e)}"}
