#!/usr/bin/env python3
"""
browser_use_search_agent.py
Documentation URL Extractor using browser-use

DOM traversal.
URL limited to prevent token overflow
"""

import asyncio
import json
import os
from typing import List
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from browser_use import Agent, Controller, ActionResult, BrowserSession, BrowserProfile

load_dotenv()


class DocsExtractor:
    def __init__(self):
        self.controller = Controller()
        self.extracted_urls: List[str] = []
        self.setup_actions()

        # To Use DeepSeek - 30X cheaper
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        if not deepseek_api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment")

        """self.llm = ChatDeepSeek(
            base_url='https://api.deepseek.com/v1',
            model='deepseek-chat',
            api_key=SecretStr(deepseek_api_key)
        )"""

        api_key = os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(
            model="gpt-4o",
            api_key=SecretStr(api_key)
        )
        #Create reusable browser profile
        self.browser_profile = BrowserProfile(
            headless=True,
            user_data_dir=None,  # Ephemeral profiles for parallel sessions
            keep_alive=True,  # Keep sessions alive for reuse
            wait_for_network_idle_page_load_time=1.0,  # Faster loading
            maximum_wait_page_load_time=8.0,  # Shorter timeout
            wait_between_actions=0.2,  # Faster actions
            highlight_elements=False,  # Disable for speed

        )

        #Session pool for reuse
        self.session_pool: List[BrowserSession] = []
        self.max_pool_size = 3
        self._pool_lock = asyncio.Lock()

    def setup_actions(self):
        @self.controller.action("Save extracted documentation URLs")
        def save_urls(urls_text: str) -> ActionResult:
            """Save the extracted URLs from navigation"""
            lines = [line.strip() for line in urls_text.strip().split('\n')]
            urls = [line for line in lines if line.startswith('http')]

            # LIMIT TO 50 URLs to prevent token overflow
            if len(urls) > 50:
                print(f"⚠️ Found {len(urls)} URLs, limiting to 50 to prevent token overflow")
                urls = urls[:50]

            self.extracted_urls = urls
            print(f"📋 Extracted {len(urls)} URLs")

            return ActionResult(
                extracted_content=f"Saved {len(urls)} URLs successfully",
                include_in_memory=True
            )

    async def get_browser_session(self) -> BrowserSession:
        """Get or create a browser session from pool"""
        async with self._pool_lock:
            if self.session_pool:
                session = self.session_pool.pop()
                print(f"♻️ Reusing browser session (pool: {len(self.session_pool)})")
                return session

        # Create new session if pool empty
        session = BrowserSession(browser_profile=self.browser_profile)
        await session.start()
        print(f"🆕 Created new browser session")
        return session

    async def return_browser_session(self, session: BrowserSession):
        """Return session to pool or close if pool full"""
        async with self._pool_lock:
            if len(self.session_pool) < self.max_pool_size:
                self.session_pool.append(session)
                print(f"↩️ Returned session to pool (pool: {len(self.session_pool)})")
            else:
                try:
                    await session.close()
                    print(f"🗑️ Closed excess session")
                except Exception as e:
                    print(f"⚠️ Error closing session: {e}")

    async def extract_docs_urls(self, query: str) -> List[str]:
        """Extract all documentation URLs"""
        browser_session = None

        task = f"""
Find documentation for: {query}

Steps:
1. Search and go to the given page or query.
2. Go to the correct site. It would mostly be tutorials pages or documentation pages. For documentation sites, Some times its like getting started links  
3. Check is this is the correct site having the full documentation we need. If not, go back. For pages like aws, they might have different kinds of documentations branched across multiple pages. So you might have to navigate across sites to get to the useful one. Useful pages we need would have the introduction, getting started, installations, setting up kind of pages for example. 
4. Find if a main navigation or sidebar for all pages and subpages exists, (usually on the left or top)
5. Extract all the links for pages and subpages in that documentation and make sure they're a part of the documentation. One strategy is that you could, extract ALL href attributes from navigation links. Another one is You can use JavaScript/DOM queries to extract ALL href attributes. But use your own techniques.
6. Convert relative URLs to absolute URLs if needed
7. Call 'Save extracted documentation URLs' with all URLs (one per line)

IMPORTANT: Extract every relevant URLs for comprehensive tutorials or documentations. Include all subsections, not just main categories.

The important goal is that we have the full complete documentation/tutorial index. This must be the documentation/tutorial someone can refer to and build stuff. 
Focus on the main documentation/tutorial pages, not blog/marketing pages.
Extract actual URLs from DOM elements, don't right-click.
"""

        agent = Agent(
            task=task,
            llm=self.llm,
            controller=self.controller,
            use_vision=False,
            browser_session=browser_session,

        )

        print(f"🔍 Extracting documentation URLs for: {query}")
        try:
            await agent.run()
        except Exception as e:
            print(f"❌ Browser agent failed: {e}")
            # Return empty list on failure instead of crashing
            return []

        return self.extracted_urls


async def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Extract documentation URLs")
    parser.add_argument("query", help="Documentation to find (e.g., 'crawl4ai documentation')")
    parser.add_argument("--output", help="Output file for URLs")

    args = parser.parse_args()

    try:
        extractor = DocsExtractor()
        urls = await extractor.extract_docs_urls(args.query)

        print(f"\n✅ Found {len(urls)} documentation URLs:")
        for i, url in enumerate(urls, 1):
            print(f"  {i:2d}. {url}")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump({"query": args.query, "urls": urls}, f, indent=2)
            print(f"\n📁 URLs saved to: {args.output}")

        return urls

    except Exception as e:
        print(f"❌ Failed: {e}")
        return []


if __name__ == "__main__":
    print("🔍 Simple Documentation URL Extractor")
    #print("Uses DeepSeek + DOM traversal (no right-clicking needed)")
    print("⚠️ Limited to 50 URLs max to prevent token overflow")

    asyncio.run(main())