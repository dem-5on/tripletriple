import httpx
from pydantic import BaseModel, Field
from .tools import Tool


class WebSearchSchema(BaseModel):
    query: str = Field(..., description="Search query string")
    num_results: int = Field(5, description="Number of results to return")


class WebSearchTool(Tool):
    """
    Web search tool using DuckDuckGo HTML (no API key required).
    For production, swap with Google Custom Search or Brave API.
    """

    name = "web_search"
    description = "Search the web for information. Returns titles, URLs, and snippets."
    args_schema = WebSearchSchema

    async def run(self, query: str, num_results: int = 5) -> str:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    "https://html.duckduckgo.com/html/",
                    params={"q": query},
                    headers={"User-Agent": "Mozilla/5.0"},
                    timeout=10,
                )
                resp.raise_for_status()

                # Simple extraction from DuckDuckGo HTML results
                from html.parser import HTMLParser

                results = []

                class DDGParser(HTMLParser):
                    in_result = False
                    current = {}

                    def handle_starttag(self, tag, attrs):
                        attrs_dict = dict(attrs)
                        if tag == "a" and "result__a" in attrs_dict.get("class", ""):
                            self.in_result = True
                            self.current = {
                                "url": attrs_dict.get("href", ""),
                                "title": "",
                                "snippet": "",
                            }

                    def handle_data(self, data):
                        if self.in_result and self.current:
                            if not self.current["title"]:
                                self.current["title"] = data.strip()
                            else:
                                self.current["snippet"] += data.strip() + " "

                    def handle_endtag(self, tag):
                        if tag == "a" and self.in_result:
                            self.in_result = False
                            if self.current.get("title"):
                                results.append(self.current)
                            self.current = {}

                parser = DDGParser()
                parser.feed(resp.text)

                if not results:
                    return f"No results found for '{query}'."

                output = f"Search results for '{query}':\n\n"
                for i, r in enumerate(results[:num_results], 1):
                    output += f"{i}. **{r['title']}**\n"
                    output += f"   URL: {r['url']}\n"
                    if r["snippet"]:
                        output += f"   {r['snippet'].strip()}\n"
                    output += "\n"

                return output

        except Exception as e:
            return f"Search error: {e}"
