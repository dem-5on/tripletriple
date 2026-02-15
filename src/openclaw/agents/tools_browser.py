import asyncio
from pydantic import BaseModel, Field
from playwright.async_api import async_playwright
from .tools import Tool

class BrowserNavigateSchema(BaseModel):
    url: str = Field(..., description="URL to navigate to")

class BrowserNavigateTool(Tool):
    name = "browser_navigate"
    description = "Navigate to a URL and capture the content."
    args_schema = BrowserNavigateSchema

    async def run(self, url: str) -> str:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            try:
                await page.goto(url)
                content = await page.content()
                # Simple text extraction for now
                text = await page.evaluate("document.body.innerText")
                return text[:5000] + "..." if len(text) > 5000 else text
            except Exception as e:
                return f"Error navigating to {url}: {e}"
            finally:
                await browser.close()
