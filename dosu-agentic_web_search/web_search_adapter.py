"""
web_search_adapter.py

"""

import os
import requests
from dotenv import load_dotenv
import time
from html import unescape
import re

load_dotenv()

BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
BRAVE_API_URL = "https://api.search.brave.com/res/v1/web/search"


def brave_search(query, count=5):
    """
    Search using Brave API and return cleaned results

    Args:
        query (str): Search query
        count (int): Number of results to return (default 5)

    Returns:
        List[Dict]: List of search results with title, url, description
    """

    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": BRAVE_API_KEY,
    }
    params = {
        "q": query,
        "count": count,
    }

    try:
        # Add small delay to be respectful to API
        time.sleep(0.5)

        response = requests.get(BRAVE_API_URL, headers=headers, params=params, timeout=30)

        if response.status_code == 200:
            results = response.json()
            links = []

            for item in results.get("web", {}).get("results", []):
                # Ensure we have required fields
                title = clean_html(item.get("title", ""))
                url = item.get("url", "")
                description = clean_html(item.get("description", ""))

                if title and url:  # Only add if we have essential data
                    links.append({
                        "title": title,
                        "url": url,
                        "description": description,
                    })

            return links

        elif response.status_code == 429:
            # Rate limit - wait and retry once
            print("⏳ Rate limited, waiting 5 seconds...")
            time.sleep(5)
            return brave_search(query, count)  # Single retry

        else:
            raise Exception(f"Brave API Error: {response.status_code} - {response.text}")

    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error calling Brave API: {str(e)}")


def clean_html(text):
    """
    Clean HTML tags and decode entities from text

    Args:
        text (str): Text potentially containing HTML

    Returns:
        str: Cleaned text
    """
    if not text:
        return ""

    # Remove HTML tags
    text = re.sub(r'<[^>]*>', '', text)

    # Decode HTML entities like &amp;, &#x27;
    text = unescape(text)

    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# Test function for standalone usage
def test_search():
    """Test function - only runs when script is executed directly"""
    test_queries = [
        "AWS S3 documentation tutorial",
        "Python programming guide",
        "React.js getting started"
    ]

    for query in test_queries:
        print(f"\n🔍 Testing query: {query}")
        try:
            results = brave_search(query, count=3)
            for i, r in enumerate(results, 1):
                print(f"  {i}. {r['title']}")
                print(f"     {r['url']}")
                print(f"     {r['description'][:100]}...")
                print()
        except Exception as e:
            print(f"❌ Error: {e}")


# Only run test when script is executed directly
if __name__ == "__main__":
    print("🧪 Testing Brave Search Adapter")
    test_search()