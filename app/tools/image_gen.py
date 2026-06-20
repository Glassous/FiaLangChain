from langchain_core.tools import tool

@tool
def generate_image(prompt: str, resolution: str = "2048x2048") -> dict:
    """根据用户描述生成/绘制图像或画图。当用户要求编写代码、生成网页、设计 HTML/CSS/JS、编写文字时，千万不要调用此工具。仅在明确要求画图、绘制或生成图像照片时使用。
    
    Args:
        prompt: Detailed text description of the image to generate
        resolution: Image resolution, e.g. '2048x2048', '2560x1440'
    """
    return {
        "status": "image generation request received"
    }
