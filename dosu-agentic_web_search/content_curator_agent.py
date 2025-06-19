#!/usr/bin/env python3
"""
content_curator_agent.py
AI Content Curator for Learnchain - Orchestrates web search, URL extraction, and content crawling
"""

import asyncio
import json
import os
import psutil
import signal
import time
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime
import uuid
from dataclasses import dataclass, asdict

from dotenv import load_dotenv
import openai
from web_search_adapter import brave_search
from browser_use_search_agent import DocsExtractor

# Simplified web crawler for specific URLs
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter

load_dotenv()


@dataclass
class ExtractionAttempt:
    """Track individual extraction attempts with timing and results"""
    site_title: str
    site_url: str
    start_time: str
    end_time: str = None
    duration_seconds: float = 0
    status: str = "running"  # running, completed, timeout, error
    urls_extracted: int = 0
    error_message: str = None
    process_terminated: bool = False


@dataclass
class CurationState:
    """Complete state management for the curation process"""
    session_id: str
    course_requirements: Dict
    search_sites: List[Dict] = None
    extracted_urls: List[Dict] = None
    approved_urls: List[Dict] = None
    crawl_results: Dict = None
    extraction_attempts: List[ExtractionAttempt] = None
    status: str = "initialized"
    created_at: str = None
    last_updated: str = None
    error_log: List[str] = None

    def __post_init__(self):
        if self.error_log is None:
            self.error_log = []
        if self.extraction_attempts is None:
            self.extraction_attempts = []


class BrowserUseWatcher:
    """Process watcher that can terminate browser-use agents based on time limits"""

    def __init__(self, timeout_minutes: int = 8):
        self.timeout_seconds = timeout_minutes * 60
        self.active_processes = {}
        self.terminated_processes = []

    async def start_extraction_with_watcher(self, docs_extractor: DocsExtractor, query: str, site_info: Dict) -> tuple:
        """Start browser-use extraction with process monitoring and timeout"""

        start_time = time.time()
        attempt = ExtractionAttempt(
            site_title=site_info['title'],
            site_url=site_info['url'],
            start_time=datetime.now().isoformat(),
            status="running"
        )

        print(f"🕐 Starting extraction with {self.timeout_seconds // 60}min timeout: {site_info['title']}")

        # Create extraction task
        extraction_task = asyncio.create_task(
            docs_extractor.extract_docs_urls(query)
        )

        # Create watcher task
        watcher_task = asyncio.create_task(
            self._watch_process(extraction_task, attempt, start_time)
        )

        try:
            # Wait for either extraction to complete or watcher to timeout
            done, pending = await asyncio.wait(
                [extraction_task, watcher_task],
                return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel any remaining tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Get results
            if extraction_task in done and not extraction_task.cancelled():
                # Extraction completed normally
                urls = extraction_task.result()
                attempt.status = "completed"
                attempt.urls_extracted = len(urls)
                attempt.end_time = datetime.now().isoformat()
                attempt.duration_seconds = time.time() - start_time
                print(f"✅ Extraction completed: {len(urls)} URLs in {attempt.duration_seconds:.1f}s")
                return urls, attempt

            elif watcher_task in done:
                # Watcher triggered timeout
                attempt.status = "timeout"
                attempt.end_time = datetime.now().isoformat()
                attempt.duration_seconds = time.time() - start_time
                attempt.process_terminated = True
                print(f"⏰ Extraction timed out after {attempt.duration_seconds:.1f}s")

                # Try to kill any remaining browser processes
                await self._cleanup_browser_processes()

                return [], attempt
            else:
                # Unexpected case
                attempt.status = "error"
                attempt.error_message = "Unexpected completion state"
                attempt.end_time = datetime.now().isoformat()
                attempt.duration_seconds = time.time() - start_time
                return [], attempt

        except Exception as e:
            # Handle extraction errors
            attempt.status = "error"
            attempt.error_message = str(e)
            attempt.end_time = datetime.now().isoformat()
            attempt.duration_seconds = time.time() - start_time
            print(f"❌ Extraction error: {e}")
            return [], attempt

    async def _watch_process(self, extraction_task: asyncio.Task, attempt: ExtractionAttempt, start_time: float):
        """Watch the extraction process and terminate if timeout exceeded"""

        check_interval = 5  # Check every 5 seconds

        while not extraction_task.done():
            elapsed = time.time() - start_time

            if elapsed >= self.timeout_seconds:
                print(f"⚠️ Timeout reached ({elapsed:.1f}s), terminating extraction...")

                # Cancel the extraction task
                extraction_task.cancel()

                # Kill browser processes
                await self._cleanup_browser_processes()

                # Mark as terminated
                self.terminated_processes.append({
                    'site': attempt.site_title,
                    'duration': elapsed,
                    'timestamp': datetime.now().isoformat()
                })

                break

            # Wait before next check
            await asyncio.sleep(check_interval)

    async def _cleanup_browser_processes(self):
        """Cleanup any lingering browser processes"""
        try:
            # Find and kill chromium/browser processes
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    proc_info = proc.info
                    if proc_info['name'] and any(browser in proc_info['name'].lower()
                                                 for browser in ['chromium', 'chrome', 'playwright']):
                        if proc_info['cmdline'] and any('browser-use' in str(cmd) or 'playwright' in str(cmd)
                                                        for cmd in proc_info['cmdline']):
                            print(f"🔪 Terminating browser process: {proc_info['pid']}")
                            proc.terminate()
                            try:
                                proc.wait(timeout=5)
                            except psutil.TimeoutExpired:
                                proc.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            print(f"⚠️ Error cleaning up processes: {e}")

    def get_summary(self) -> Dict:
        """Get summary of all extraction attempts"""
        return {
            'total_attempts': len(self.terminated_processes),
            'timeout_count': len([p for p in self.terminated_processes]),
            'terminated_processes': self.terminated_processes
        }


class ContentCuratorAgent:
    """Main brain agent orchestrating the content curation pipeline"""

    def __init__(self):
        # Check for required API keys
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError("❌ OPENAI_API_KEY not found in environment variables")

        self.openai_client = openai.OpenAI(api_key=openai_key)
        self.docs_extractor = DocsExtractor()
        self.browser_watcher = BrowserUseWatcher(timeout_minutes=8)
        self.state: Optional[CurationState] = None
        self.output_dir = "curated_content"

        # Test OpenAI connection
        self._test_openai_connection()

    def _test_openai_connection(self):
        """Test OpenAI API connection"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Return just the word 'test'"}],
                max_tokens=10,
                temperature=0
            )
            result = response.choices[0].message.content.strip()
            print(f"🔗 Connection: ✅ ({result})")
        except Exception as e:
            print(f"❌ LLM Connection Failed: {e}")
            raise ValueError(f"OpenAI API test failed: {e}")

    async def start_curation(self, requirements_file: str) -> str:
        """Initialize curation from course requirements"""
        with open(requirements_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        session_id = str(uuid.uuid4())[:8]
        self.state = CurationState(
            session_id=session_id,
            course_requirements=data,
            created_at=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat()
        )

        # Create session directory with robust error handling
        try:
            session_dir = Path(f"{self.output_dir}/{session_id}")
            session_dir.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created session directory: {session_dir}")

            # Test write access
            test_file = session_dir / "test.txt"
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write("test")
            test_file.unlink()  # Delete test file

        except Exception as e:
            print(f"❌ Failed to create session directory: {e}")
            raise ValueError(f"Cannot create session directory: {e}")

        self._save_state()

        print(f"🧠 Content Curator Started")
        print(f"📋 Course: {data['course_requirements']['training_prompt']}")
        print(f"🆔 Session: {session_id}")

        return session_id

    async def execute_search_strategy(self) -> List[Dict]:
        """Brain plans and executes web search strategy"""
        print("🧠 Planning search strategy...")

        try:
            # Generate search queries using enhanced brain LLM
            course_req = self.state.course_requirements["course_requirements"]
            search_instructions = self.state.course_requirements.get("web_search_instructions", {})

            queries = await self._generate_search_queries(course_req, search_instructions)

            # Execute searches
            all_sites = []
            for query in queries:
                print(f"🔍 Searching: {query}")
                results = brave_search(query, count=5)
                all_sites.extend(results)

            # Brain filters to top 5 sites
            filtered_sites = await self._filter_best_sites(all_sites, course_req)

            self.state.search_sites = filtered_sites
            self.state.status = "sites_found"
            self.state.last_updated = datetime.now().isoformat()
            self._save_state()

            print(f"✅ Selected {len(filtered_sites)} top sites")
            return filtered_sites

        except Exception as e:
            error_msg = f"Search strategy failed: {e}"
            self.state.error_log.append(error_msg)
            self.state.last_updated = datetime.now().isoformat()
            self._save_state()
            print(f"❌ {error_msg}")
            raise

    async def extract_all_urls(self) -> List[Dict]:
        """Extract comprehensive URL list using browser-use agents with process watcher"""
        print("🤖 Extracting documentation URLs with process monitoring...")

        all_urls = []

        for site in self.state.search_sites:
            site_type = self._classify_site_type(site)

            if site_type == "documentation":
                print(f"📚 Extracting docs from: {site['title']}")

                # Use browser-use watcher for monitored extraction
                doc_urls, attempt = await self.browser_watcher.start_extraction_with_watcher(
                    self.docs_extractor,
                    f"documentation for {site['url']}",
                    site
                )

                # Store attempt in brain state
                self.state.extraction_attempts.append(attempt)

                if attempt.status == "completed" and doc_urls:
                    # Limit to prevent token overflow
                    doc_urls = doc_urls[:50]

                    for url in doc_urls:
                        all_urls.append({
                            "url": url,
                            "title": self._extract_title_from_url(url),
                            "source": site['title'],
                            "type": "documentation",
                            "priority": 1,
                            "extraction_method": "browser-use-success"
                        })

                    print(f"✅ Successfully extracted {len(doc_urls)} URLs")

                elif attempt.status == "timeout":
                    # Fallback to main URL on timeout
                    all_urls.append({
                        "url": site['url'],
                        "title": site['title'],
                        "source": site['title'],
                        "type": "main_page",
                        "priority": 2,
                        "extraction_method": "timeout-fallback"
                    })

                    error_msg = f"Browser-use timeout for {site['title']} after {attempt.duration_seconds:.1f}s"
                    self.state.error_log.append(error_msg)
                    print(f"⏰ {error_msg}")

                elif attempt.status == "error":
                    # Fallback to main URL on error
                    all_urls.append({
                        "url": site['url'],
                        "title": site['title'],
                        "source": site['title'],
                        "type": "main_page",
                        "priority": 2,
                        "extraction_method": "error-fallback"
                    })

                    error_msg = f"Browser-use error for {site['title']}: {attempt.error_message}"
                    self.state.error_log.append(error_msg)
                    print(f"❌ {error_msg}")

            else:
                # For tutorial/guide sites, add main URL
                all_urls.append({
                    "url": site['url'],
                    "title": site['title'],
                    "source": site['title'],
                    "type": site_type,
                    "priority": 2,
                    "extraction_method": "direct-url"
                })
                print(f"✅ Added {site_type}: {site['title']}")

        # Update brain state with extraction summary
        watcher_summary = self.browser_watcher.get_summary()
        self.state.error_log.append(
            f"Extraction summary: {len(self.state.extraction_attempts)} attempts, {watcher_summary['timeout_count']} timeouts")

        try:
            # Brain organizes and enhances URLs
            organized_urls = await self._organize_and_rank_urls(all_urls)

            self.state.extracted_urls = organized_urls
            self.state.status = "urls_extracted"
            self.state.last_updated = datetime.now().isoformat()
            self._save_state()

            print(f"✅ Organized {len(organized_urls)} URLs")

            # Print extraction summary
            self._print_extraction_summary()

            return organized_urls

        except Exception as e:
            error_msg = f"URL organization failed: {e}"
            self.state.error_log.append(error_msg)
            self.state.last_updated = datetime.now().isoformat()
            self._save_state()
            print(f"❌ {error_msg}")
            # Return raw URLs if organization fails
            self.state.extracted_urls = all_urls
            return all_urls

    def _print_extraction_summary(self):
        """Print detailed summary of extraction attempts"""
        print(f"\n📊 Extraction Summary:")

        successful = [a for a in self.state.extraction_attempts if a.status == "completed"]
        timeouts = [a for a in self.state.extraction_attempts if a.status == "timeout"]
        errors = [a for a in self.state.extraction_attempts if a.status == "error"]

        print(f"   ✅ Successful: {len(successful)}")
        print(f"   ⏰ Timeouts: {len(timeouts)}")
        print(f"   ❌ Errors: {len(errors)}")

        if timeouts:
            print(f"\n   Timed out sites:")
            for attempt in timeouts:
                print(f"     • {attempt.site_title} ({attempt.duration_seconds:.1f}s)")

        if errors:
            print(f"\n   Error sites:")
            for attempt in errors:
                print(f"     • {attempt.site_title}: {attempt.error_message}")

    def create_approval_table(self) -> str:
        """Generate structured table for trainer approval"""
        print("📋 Creating approval table...")

        try:
            # Ensure directory exists
            session_dir = Path(f"{self.output_dir}/{self.state.session_id}")
            session_dir.mkdir(parents=True, exist_ok=True)

            # Create approval table data
            table_data = []
            for i, url_info in enumerate(self.state.extracted_urls, 1):
                table_data.append({
                    "ID": i,
                    "Title": url_info["title"][:60] + "..." if len(url_info["title"]) > 60 else url_info["title"],
                    "URL": url_info["url"],
                    "Source": url_info["source"],
                    "Type": url_info["type"],
                    "Priority": url_info["priority"],
                    "Relevance": url_info.get("relevance", "Medium"),
                    "Method": url_info.get("extraction_method", "unknown")
                })

            # Save table for trainer review
            table_file = session_dir / "approval_table.json"
            with open(table_file, 'w', encoding='utf-8') as f:
                json.dump(table_data, f, indent=2)

            # Create readable markdown table
            markdown_table = self._format_markdown_table(table_data)

            # Save markdown version with extraction summary
            md_file = session_dir / "approval_table.md"
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write("# Content Approval Table\n\n")
                f.write(f"**Course:** {self.state.course_requirements['course_requirements']['training_prompt']}\n\n")

                # Add extraction summary
                f.write("## Extraction Summary\n\n")
                successful = len([a for a in self.state.extraction_attempts if a.status == "completed"])
                timeouts = len([a for a in self.state.extraction_attempts if a.status == "timeout"])
                errors = len([a for a in self.state.extraction_attempts if a.status == "error"])
                f.write(f"- **Successful extractions:** {successful}\n")
                f.write(f"- **Timeouts:** {timeouts}\n")
                f.write(f"- **Errors:** {errors}\n\n")

                f.write("## URLs for Approval\n\n")
                f.write(markdown_table)

            self.state.last_updated = datetime.now().isoformat()
            self._save_state()

            print(f"📋 Approval table saved: {md_file}")
            print("\n" + markdown_table)

            return str(table_file)

        except Exception as e:
            error_msg = f"Failed to create approval table: {e}"
            self.state.error_log.append(error_msg)
            self.state.last_updated = datetime.now().isoformat()
            self._save_state()
            print(f"❌ {error_msg}")
            raise

    async def process_approval(self, approved_ids: List[int]) -> None:
        """Process trainer approval and prepare for crawling"""
        try:
            approved_urls = []
            for i, url_info in enumerate(self.state.extracted_urls, 1):
                if i in approved_ids:
                    approved_urls.append(url_info)

            self.state.approved_urls = approved_urls
            self.state.status = "approved"
            self.state.last_updated = datetime.now().isoformat()
            self._save_state()

            print(f"✅ {len(approved_urls)} URLs approved for crawling")

        except Exception as e:
            error_msg = f"Failed to process approval: {e}"
            self.state.error_log.append(error_msg)
            self.state.last_updated = datetime.now().isoformat()
            self._save_state()
            print(f"❌ {error_msg}")
            raise

    async def crawl_approved_content(self) -> Dict:
        """Crawl content from approved URLs using simplified crawler"""
        print("🕷️ Starting content crawling...")

        try:
            # Ensure directories exist
            session_dir = Path(f"{self.output_dir}/{self.state.session_id}")
            crawl_dir = session_dir / "content"
            crawl_dir.mkdir(parents=True, exist_ok=True)

            # Configure browser
            browser_config = BrowserConfig(
                browser_type="chromium",
                headless=True,
                verbose=False
            )

            crawler_config = CrawlerRunConfig(
                word_count_threshold=10,
                excluded_tags=["script", "style", "nav", "header", "footer"],
                page_timeout=30000,
                cache_mode=CacheMode.BYPASS,
                markdown_generator=DefaultMarkdownGenerator(
                    content_filter=PruningContentFilter(threshold=0.1)
                )
            )

            # Parallel crawling with semaphore
            semaphore = asyncio.Semaphore(5)  # 5 concurrent crawls
            results = []

            async def crawl_single_url(url_info):
                async with semaphore:
                    try:
                        async with AsyncWebCrawler(config=browser_config) as crawler:
                            result = await crawler.arun(url_info["url"], config=crawler_config)

                            if result.success:
                                # Extract content
                                content = ""
                                if hasattr(result, 'markdown') and result.markdown:
                                    content = result.markdown.fit_markdown or result.markdown.raw_markdown or ""

                                # Save content file
                                safe_title = "".join(
                                    c for c in url_info["title"] if c.isalnum() or c in (' ', '-', '_')).rstrip()[:50]
                                file_path = f"{crawl_dir}/{safe_title}_{hash(url_info['url']) % 10000}.md"

                                with open(file_path, 'w', encoding='utf-8') as f:
                                    f.write(f"# {url_info['title']}\n\n")
                                    f.write(f"**Source:** {url_info['source']}\n")
                                    f.write(f"**URL:** {url_info['url']}\n")
                                    f.write(f"**Type:** {url_info['type']}\n")
                                    f.write(
                                        f"**Extraction Method:** {url_info.get('extraction_method', 'unknown')}\n\n")
                                    f.write("---\n\n")
                                    f.write(content)

                                word_count = len(content.split()) if content else 0

                                results.append({
                                    "url": url_info["url"],
                                    "title": url_info["title"],
                                    "success": True,
                                    "word_count": word_count,
                                    "file_path": file_path,
                                    "extraction_method": url_info.get('extraction_method', 'unknown')
                                })

                                print(f"✅ {url_info['title'][:40]:<40} | {word_count:>5} words")
                            else:
                                results.append({
                                    "url": url_info["url"],
                                    "title": url_info["title"],
                                    "success": False,
                                    "error": "Crawl failed",
                                    "extraction_method": url_info.get('extraction_method', 'unknown')
                                })
                                print(f"❌ {url_info['title'][:40]:<40} | Failed")

                    except Exception as e:
                        results.append({
                            "url": url_info["url"],
                            "title": url_info["title"],
                            "success": False,
                            "error": str(e),
                            "extraction_method": url_info.get('extraction_method', 'unknown')
                        })
                        print(f"❌ {url_info['title'][:40]:<40} | Error: {e}")

            # Execute all crawls in parallel
            tasks = [crawl_single_url(url_info) for url_info in self.state.approved_urls]
            await asyncio.gather(*tasks, return_exceptions=True)

            # Create summary
            successful = [r for r in results if r["success"]]
            total_words = sum(r.get("word_count", 0) for r in successful)

            crawl_summary = {
                "total_urls": len(self.state.approved_urls),
                "successful": len(successful),
                "failed": len(results) - len(successful),
                "total_words": total_words,
                "avg_words_per_page": total_words // max(1, len(successful)),
                "crawl_timestamp": datetime.now().isoformat(),
                "extraction_summary": {
                    "total_attempts": len(self.state.extraction_attempts),
                    "successful_extractions": len(
                        [a for a in self.state.extraction_attempts if a.status == "completed"]),
                    "timeout_extractions": len([a for a in self.state.extraction_attempts if a.status == "timeout"]),
                    "error_extractions": len([a for a in self.state.extraction_attempts if a.status == "error"])
                },
                "results": results
            }

            # Save summary
            summary_file = session_dir / "crawl_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(crawl_summary, f, indent=2)

            self.state.crawl_results = crawl_summary
            self.state.status = "completed"
            self.state.last_updated = datetime.now().isoformat()
            self._save_state()

            print(f"\n🎉 Crawling Complete!")
            print(f"📊 Success: {len(successful)}/{len(self.state.approved_urls)} URLs")
            print(f"📄 Content: {total_words:,} total words")
            print(f"📁 Saved to: {crawl_dir}")

            return crawl_summary

        except Exception as e:
            error_msg = f"Crawling failed: {e}"
            self.state.error_log.append(error_msg)
            self.state.last_updated = datetime.now().isoformat()
            self._save_state()
            print(f"❌ {error_msg}")
            raise

    # Enhanced Brain LLM helper methods
    async def _generate_search_queries(self, course_req: Dict, search_instructions: Dict) -> List[str]:
        """Generate targeted search queries using enhanced brain LLM"""

        prompt = f"""You are an AI Content Curator planning a comprehensive search strategy for technical training content.

MISSION: Find the best learning resources for backend engineers who need practical, hands-on training.

COURSE DETAILS:
- Training Objective: {course_req['training_prompt']}
- Target Audience: {course_req['job_description']}
- Learning Context: {search_instructions.get('target_content', 'Professional technical training')}

CONTENT REQUIREMENTS:
1. OFFICIAL DOCUMENTATION - Primary authoritative sources (APIs, guides, references)
2. PRACTICAL TUTORIALS - Step-by-step implementation guides with real examples
3. BEST PRACTICES - Industry standards, security patterns, optimization techniques
4. REAL-WORLD EXAMPLES - Case studies, production patterns, common pitfalls
5. UNIQUE/INTERESTING CONTENT - Advanced techniques, lesser-known features, expert insights

SEARCH STRATEGY:
- Prioritize official docs and established technical sites
- Include hands-on tutorials with code examples
- Find content that covers both basics AND advanced topics
- Look for practical implementation patterns
- Discover interesting/unique aspects that make training engaging

Generate 4-5 focused search queries that will find comprehensive learning content.
Each query should target different aspects: official docs, tutorials, best practices, examples, and unique insights.

Return ONLY a JSON array of search query strings, nothing else.
Example format: ["query 1", "query 2", "query 3", "query 4"]"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=300

            )

            content = response.choices[0].message.content.strip()
            print(f"🤖 Planned Search Queries: {content}")

            # Clean response
            if content.startswith('```'):
                lines = content.split('\n')
                if lines[0].startswith('```'):
                    lines = lines[1:]
                if lines[-1].strip() == '```':
                    lines = lines[:-1]
                content = '\n'.join(lines).strip()

            queries = json.loads(content)
            print(f"🧹 Cleaned Response: {queries}")

            if isinstance(queries, list) and len(queries) > 0:
                return queries
            else:
                raise ValueError("Invalid query format returned")

        except (json.JSONDecodeError, Exception) as e:
            error_msg = f"LLM search query generation failed: {e}"
            self.state.error_log.append(error_msg)
            print(f"❌ {error_msg}")
            # Graceful failure - use simple generic query
            topic = course_req['training_prompt'].split()[:3]
            return [" ".join(topic) + " documentation"]

    async def _filter_best_sites(self, sites: List[Dict], course_req: Dict) -> List[Dict]:
        """Filter to top 5 most relevant sites using enhanced criteria"""

        prompt = f"""You are an expert content curator selecting the BEST learning resources for technical training.

TRAINING OBJECTIVE: {course_req['training_prompt']}
TARGET LEARNERS: {course_req['job_description']}

EVALUATION CRITERIA (in priority order):
1. AUTHORITY - Official docs, established tech companies, recognized experts
2. PRACTICALITY - Hands-on tutorials, real implementation examples, working code
3. COMPREHENSIVENESS - Covers basics to advanced, end-to-end workflows
4. RELEVANCE - Directly applicable to the training objective
5. QUALITY - Well-structured, up-to-date, professional content

AVAILABLE SITES:
{json.dumps(sites[:15], indent=2)}

Select the TOP 5 sites that will provide the most valuable learning content.
Prioritize: Official documentation > Quality tutorials > Practical guides > Expert blogs

Return ONLY a JSON array of the 5 best sites with their original structure.
Example format: [{{"title": "...", "url": "...", "description": "..."}}, ...]"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )

            content = response.choices[0].message.content.strip()

            # Clean markdown code blocks if present
            if content.startswith('```'):
                lines = content.split('\n')
                if lines[0].startswith('```'):
                    lines = lines[1:]
                if lines[-1].strip() == '```':
                    lines = lines[:-1]
                content = '\n'.join(lines).strip()

            filtered = json.loads(content)

            if isinstance(filtered, list):
                return filtered[:5]
            else:
                raise ValueError("Invalid response format")

        except (json.JSONDecodeError, Exception) as e:
            error_msg = f"Site filtering failed: {e}"
            self.state.error_log.append(error_msg)
            print(f"❌ {error_msg}")
            return sites[:5]

    async def _organize_and_rank_urls(self, urls: List[Dict]) -> List[Dict]:
        """Organize URLs with enhanced relevance scoring and prioritization"""

        prompt = f"""You are an expert learning architect organizing content for technical training.

TRAINING GOAL: {self.state.course_requirements['course_requirements']['training_prompt']}
TARGET AUDIENCE: {self.state.course_requirements['course_requirements']['job_description']}

CONTENT EVALUATION FRAMEWORK:
- HIGH relevance: Direct implementation guides, official docs, core concepts
- MEDIUM relevance: Supporting tutorials, best practices, related examples  
- LOW relevance: General information, tangential topics, basic overviews

PRIORITY SCORING (1=highest, 5=lowest):
- Priority 1: Essential official docs, getting started guides, core APIs
- Priority 2: Implementation tutorials, practical examples, integration guides
- Priority 3: Best practices, security guides, optimization techniques
- Priority 4: Advanced topics, specialized use cases, expert insights
- Priority 5: General resources, background information, supplementary content

CONTENT URLS TO ORGANIZE:
{json.dumps(urls[:100], indent=2)}

TASKS:
1. Enhance each URL with accurate relevance ("High", "Medium", "Low")
2. Assign priority (1-5) based on learning importance
3. Improve titles to be more descriptive if needed
4. Sort by learning progression: foundations → implementation → advanced

Return ONLY a JSON array of enhanced and prioritized URLs.
Example format: [{{"url": "...", "title": "...", "source": "...", "type": "...", "priority": 1, "relevance": "High"}}, ...]"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=4500,
            )

            content = response.choices[0].message.content.strip()

            # Clean markdown code blocks if present
            if content.startswith('```'):
                lines = content.split('\n')
                if lines[0].startswith('```'):
                    lines = lines[1:]
                if lines[-1].strip() == '```':
                    lines = lines[:-1]
                content = '\n'.join(lines).strip()

            enhanced = json.loads(content)

            if isinstance(enhanced, list):
                return sorted(enhanced, key=lambda x: (x.get("priority", 3), x.get("relevance") != "High"))
            else:
                raise ValueError("Invalid response format")

        except (json.JSONDecodeError, Exception) as e:
            error_msg = f"URL organization failed: {e}"
            self.state.error_log.append(error_msg)
            print(f"❌ {error_msg}")
            # Fallback to basic priority assignment
            for i, url in enumerate(urls):
                url["relevance"] = "Medium"
                url["priority"] = min(3, i // 10 + 1)
            return urls

    # Utility methods
    def _classify_site_type(self, site: Dict) -> str:
        """Classify site type for extraction strategy"""
        url_text = f"{site['url']} {site['title']}".lower()

        if any(term in url_text for term in ["docs", "documentation", "api", "reference"]):
            return "documentation"
        elif any(term in url_text for term in ["tutorial", "guide", "learn"]):
            return "tutorial"
        else:
            return "resource"

    def _extract_title_from_url(self, url: str) -> str:
        """Extract readable title from URL"""
        parts = url.strip('/').split('/')
        return parts[-1].replace('-', ' ').replace('_', ' ').title() if parts else "Page"

    def _format_markdown_table(self, data: List[Dict]) -> str:
        """Format data as markdown table"""
        if not data:
            return "No data available."

        headers = list(data[0].keys())
        header_row = "| " + " | ".join(headers) + " |"
        separator = "|" + "|".join([" --- " for _ in headers]) + "|"

        rows = []
        for item in data:
            row_values = [str(item.get(h, "")) for h in headers]
            rows.append("| " + " | ".join(row_values) + " |")

        return "\n".join([header_row, separator] + rows)

    def _save_state(self):
        """Save current state with error handling"""
        try:
            # Ensure directory exists before saving
            session_dir = Path(f"{self.output_dir}/{self.state.session_id}")
            session_dir.mkdir(parents=True, exist_ok=True)

            state_file = session_dir / "state.json"

            # Convert extraction attempts to dict for JSON serialization
            state_dict = asdict(self.state)

            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state_dict, f, indent=2)
        except Exception as e:
            print(f"⚠️ Failed to save state: {e}")

    def load_state(self, session_id: str):
        """Load existing state with error handling"""
        try:
            state_file = Path(f"{self.output_dir}/{session_id}/state.json")
            if not state_file.exists():
                raise FileNotFoundError(f"State file not found: {state_file}")

            with open(state_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Convert extraction attempts back to dataclass objects
            if 'extraction_attempts' in data and data['extraction_attempts']:
                data['extraction_attempts'] = [
                    ExtractionAttempt(**attempt)
                    for attempt in data['extraction_attempts']
                ]

            self.state = CurationState(**data)
            print(f"🔄 Resumed session: {session_id}")
            print(f"📊 Status: {self.state.status}")
            if self.state.error_log:
                print(f"⚠️ Previous errors: {len(self.state.error_log)}")
            if self.state.extraction_attempts:
                print(f"📈 Previous extractions: {len(self.state.extraction_attempts)} attempts")
        except Exception as e:
            print(f"❌ Failed to load state: {e}")
            raise


# Main execution interface
async def main():
    import argparse

    parser = argparse.ArgumentParser(description="AI Content Curator for Learnchain")
    parser.add_argument("requirements", help="course_requirements.json file")
    parser.add_argument("--session-id", help="Resume existing session")
    parser.add_argument("--auto-approve", action="store_true", help="Auto-approve all URLs")

    args = parser.parse_args()

    curator = ContentCuratorAgent()

    try:
        # Initialize or resume session
        if args.session_id:
            curator.load_state(args.session_id)
            session_id = args.session_id
        else:
            session_id = await curator.start_curation(args.requirements)

        # Execute pipeline based on current state
        if curator.state.status == "initialized":
            await curator.execute_search_strategy()

        if curator.state.status == "sites_found":
            await curator.extract_all_urls()

        if curator.state.status == "urls_extracted":
            table_file = curator.create_approval_table()

            if args.auto_approve:
                # Auto-approve all URLs
                all_ids = list(range(1, len(curator.state.extracted_urls) + 1))
                await curator.process_approval(all_ids)
            else:
                # Manual approval with include/exclude options
                print(f"\n📋 Review approval table and choose your approval method:")
                print(f"📊 Total URLs found: {len(curator.state.extracted_urls)}")
                print(f"\nOptions:")
                print(f"  1. Include specific IDs (enter IDs you want to KEEP)")
                print(f"  2. Exclude specific IDs (enter IDs you want to REMOVE)")
                print(f"  3. Approve all URLs")
                print(f"  4. Exit without approving")

                while True:
                    choice = input("\nChoose option (1-4): ").strip()

                    if choice == "1":
                        # Include mode (original behavior)
                        approved_input = input("Enter IDs to INCLUDE (comma-separated): ").strip()
                        if approved_input:
                            try:
                                approved_ids = [int(x.strip()) for x in approved_input.split(",")]
                                # Validate IDs are in range
                                max_id = len(curator.state.extracted_urls)
                                invalid_ids = [id for id in approved_ids if id < 1 or id > max_id]
                                if invalid_ids:
                                    print(f"❌ Invalid IDs: {invalid_ids}. Valid range: 1-{max_id}")
                                    continue

                                print(f"✅ Including {len(approved_ids)} URLs")
                                await curator.process_approval(approved_ids)
                                break
                            except ValueError:
                                print("❌ Invalid input. Please enter numbers separated by commas.")
                                continue
                        else:
                            print("❌ No IDs entered.")
                            continue

                    elif choice == "2":
                        # Exclude mode (new feature)
                        exclude_input = input("Enter IDs to EXCLUDE (comma-separated): ").strip()
                        if exclude_input:
                            try:
                                exclude_ids = [int(x.strip()) for x in exclude_input.split(",")]
                                # Validate IDs are in range
                                max_id = len(curator.state.extracted_urls)
                                invalid_ids = [id for id in exclude_ids if id < 1 or id > max_id]
                                if invalid_ids:
                                    print(f"❌ Invalid IDs: {invalid_ids}. Valid range: 1-{max_id}")
                                    continue

                                # Calculate approved IDs by excluding the specified ones
                                all_ids = set(range(1, len(curator.state.extracted_urls) + 1))
                                approved_ids = list(all_ids - set(exclude_ids))
                                approved_ids.sort()  # Keep them in order

                                print(f"✅ Excluding {len(exclude_ids)} URLs, approving {len(approved_ids)} URLs")
                                await curator.process_approval(approved_ids)
                                break
                            except ValueError:
                                print("❌ Invalid input. Please enter numbers separated by commas.")
                                continue
                        else:
                            print("❌ No IDs entered.")
                            continue

                    elif choice == "3":
                        # Approve all
                        all_ids = list(range(1, len(curator.state.extracted_urls) + 1))
                        print(f"✅ Approving all {len(all_ids)} URLs")
                        await curator.process_approval(all_ids)
                        break

                    elif choice == "4":
                        # Exit
                        print("❌ No URLs approved. Exiting.")
                        return

                    else:
                        print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
                        continue

        if curator.state.status == "approved":
            await curator.crawl_approved_content()

        if curator.state.error_log:
            print(f"\n⚠️ Session completed with {len(curator.state.error_log)} warnings/errors")
            for error in curator.state.error_log[-3:]:  # Show last 3 errors
                print(f"   • {error}")

        print(f"\n🎉 Content curation completed for session: {session_id}")

    except KeyboardInterrupt:
        print(f"\n⏸️ Session saved. Resume with: --session-id {session_id}")
    except Exception as e:
        print(f"❌ Error: {e}")
        if curator.state:
            curator.state.error_log.append(f"Fatal error: {e}")
            curator._save_state()
        raise


if __name__ == "__main__":
    # Proper asyncio cleanup
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏸️ Interrupted by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")