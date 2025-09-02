#!/usr/bin/env python3
"""
Debug script to understand Understat's HTML structure
"""
import requests
import re
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils import setup_logging

def debug_understat_page():
    """Debug Understat page structure"""
    logger = setup_logging()
    logger.info("🔍 DEBUGGING UNDERSTAT PAGE STRUCTURE")
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    
    # Test with 2023-2024 season
    url = "https://understat.com/league/EPL/2023"
    
    try:
        response = session.get(url)
        response.raise_for_status()
        
        html_content = response.text
        
        logger.info(f"✅ Successfully fetched page: {len(html_content)} characters")
        
        # Look for different possible patterns
        patterns = [
            r"var datesData = JSON\.parse\('(.+?)'\);",
            r"var datesData = '(.+?)';",
            r"datesData = JSON\.parse\('(.+?)'\)",
            r"datesData = '(.+?)'",
            r'"dates":\s*(\[.+?\])',
            r'var\s+matches\s*=\s*JSON\.parse\(\'(.+?)\'\)'
        ]
        
        for i, pattern in enumerate(patterns, 1):
            logger.info(f"\n🔍 Testing pattern {i}: {pattern}")
            match = re.search(pattern, html_content, re.DOTALL)
            
            if match:
                logger.info(f"✅ Pattern {i} MATCHED!")
                matched_content = match.group(1)[:200]  # First 200 chars
                logger.info(f"Sample content: {matched_content}")
            else:
                logger.info(f"❌ Pattern {i} no match")
        
        # Look for any JSON-like structures
        logger.info(f"\n🔍 Searching for JSON structures...")
        json_patterns = [
            r'JSON\.parse\(\'([^\']+)\'\)',
            r'"datetime":\s*"([^"]+)"',
            r'"xG":\s*{[^}]+}',
            r'var\s+\w+\s*=\s*\[.*?\];'
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, html_content)
            if matches:
                logger.info(f"Found {len(matches)} matches for: {pattern}")
                if len(matches) > 0:
                    logger.info(f"Sample: {str(matches[0])[:100]}")
        
        # Save a sample of the HTML for manual inspection
        os.makedirs('debug', exist_ok=True)
        with open('debug/understat_sample.html', 'w', encoding='utf-8') as f:
            f.write(html_content[:10000])  # First 10k characters
        
        logger.info(f"💾 Sample HTML saved to debug/understat_sample.html")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    debug_understat_page()