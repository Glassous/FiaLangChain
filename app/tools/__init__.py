from app.tools.calculator import calculator
from app.tools.get_time import get_time
from app.tools.weather import weather
from app.tools.web_search import web_search
from app.tools.image_gen import generate_image

# Export tools list
ALL_TOOLS = [
    calculator,
    get_time,
    weather,
    web_search,
    generate_image
]

# Map tool name to tool object
TOOL_MAP = {tool.name: tool for tool in ALL_TOOLS}
