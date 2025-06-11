import requests
from bs4 import BeautifulSoup
import time
import json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import urllib3
import os
import re

# --- [Previous code: Imports, SSL warning disable, Output dir, Selenium options] ---
# SSL 경고 무시
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 저장 경로 설정
OUTPUT_DIR = "cambridge_data_refined" # Use a new directory for refined results
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Selenium 설정
chrome_options = Options()
chrome_options.add_argument('--headless') # Run headless for efficiency
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
chrome_options.add_argument('--ignore-certificate-errors')
chrome_options.add_argument('--ignore-ssl-errors=yes')
chrome_options.add_argument("--disable-blink-features=AutomationControlled")
chrome_options.add_argument(
    "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
)
chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
chrome_options.add_experimental_option('useAutomationExtension', False)

# Initialize WebDriver
try:
    # Try using Service object (recommended way)
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
except Exception as e_service:
    print(f"WebDriver Manager failed: {e_service}")
    try:
        # Fallback to direct path if needed (adjust path if necessary)
        # driver = webdriver.Chrome(executable_path='/path/to/chromedriver', options=chrome_options)
        # Or simply let Selenium try its default behavior
        print("Falling back to default WebDriver initialization.")
        driver = webdriver.Chrome(options=chrome_options)
    except Exception as e_fallback:
        print(f"Default WebDriver initialization failed: {e_fallback}")
        # Handle the error appropriately, e.g., exit or raise
        raise Exception("Could not initialize WebDriver.") from e_fallback


driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
# --- [End of Selenium Setup] ---


# 각 URL에서 문법 페이지 직접 크롤링 (개선된 버전)
def crawl_grammar_page(url):
    try:
        driver.get(url)
        # Use WebDriverWait for more reliable loading instead of fixed sleep
        wait = WebDriverWait(driver, 10) # Wait up to 10 seconds

        # Wait for a potential main content container element to be present
        # Adjust the selector based on actual page structure inspection
        # Common Cambridge Dictionary content containers: .di-body, .entry-body, #content .definition
        try:
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".di-body, .entry-body, .cdo-topic")))
            print(f"Content indicator found for: {url}")
        except Exception as e_wait:
            print(f"Timeout waiting for main content element on {url}: {e_wait}")
            # Even if specific element isn't found, proceed to parse what's loaded
            time.sleep(3) # Fallback sleep

        page_source = driver.page_source

        # 페이지 HTML 저장 (디버깅용)
        page_name = url.split('/')[-1] or "index"
        # Sanitize page_name for filename
        page_name = re.sub(r'[\\/*?:"<>|]', "_", page_name)
        html_path = os.path.join(OUTPUT_DIR, f"{page_name}.html")
        try:
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(page_source)
            print(f"HTML saved: {html_path}")
        except Exception as e_save:
            print(f"Error saving HTML for {url}: {e_save}")


        soup = BeautifulSoup(page_source, "html.parser")

        # --- 제목 찾기 (기존 방식 유지 또는 개선) ---
        title = None
        # Try more specific title elements first
        title_selectors = [
            "h1.di-title", # More specific H1
            ".cdo-page-title",
            ".hw", # Headword class
            ".headword",
            "h1",
            ".di-title"
        ]
        for selector in title_selectors:
             title_element = soup.select_one(selector)
             if title_element and title_element.text.strip():
                 title = title_element.text.strip()
                 break

        if not title:
            title = url.split("/")[-1].replace("-", " ").title() or "Cambridge Grammar"
        print(f"Title found: '{title}'")

        # --- 콘텐츠 추출 (개선된 방식) ---
        content = ""
        content_container = None

        # 1. 가장 구체적인 컨테이너 시도 (실제 구조 확인 필요)
        #    These selectors are based on common patterns in Cambridge Dictionary.
        #    Inspect the HTML of target pages (like adjectives-and-adverbs) to confirm/refine.
        possible_selectors = [
            ".cdo-topic .di-body", # Often the main explanation area
            ".di-body",            # General definition body
            ".entry-body",         # Another common container
            ".cdo-section-body",   # Broader section body
            # Add more specific selectors if identified via inspection
        ]

        for selector in possible_selectors:
            content_container = soup.select_one(selector)
            if content_container:
                print(f"Found content container with selector: '{selector}'")
                break # Use the first one found

        if content_container:
            # Extract text, trying to preserve paragraphs
            # Remove potentially unwanted elements *before* getting text
            for unwanted in content_container.select('.exa, .examp, .xrefs, .extra_examps, .irreg-verbs, script, style, noscript, .header, .footer, .usg'):
                 unwanted.decompose() # Remove examples, cross-references, scripts etc.

            # Get text content, preserving line breaks between block elements
            content = content_container.get_text(separator='\n', strip=True)

            # Further clean the extracted text
            content = re.sub(r'\s*\n\s*', '\n', content).strip() # Normalize line breaks
             # Remove common boilerplate text patterns (add more as needed)
            boilerplate = [
                r"See also:",
                r"From English Grammar Today",
                r"Learner example:",
                r"Warning:",
                r"Compare:",
                r"© Cambridge University Press & Assessment \d{4}", # Copyright
                # Add specific phrases observed in unwanted content
            ]
            for pattern in boilerplate:
                 content = re.sub(pattern, '', content, flags=re.IGNORECASE).strip()

            print(f"Raw content length: {len(content)}")

        else:
            print(f"Could not find specific content container for {url}. Falling back to <p> tags.")
             # Fallback: Extract meaningful paragraphs from the whole page (less reliable)
            paragraphs = soup.select("p")
            content_list = [p.text.strip() for p in paragraphs if p.text.strip() and len(p.text.strip()) > 50 and not p.find_parent(class_='footer')] # Avoid footer paragraphs
            content = "\n\n".join(content_list)
            content = re.sub(r'\s+', ' ', content)
            content = re.sub(r'\n\s*\n', '\n\n', content)

        # --- 관련 문법 페이지 링크 찾기 (컨테이너 내부에서 찾아보기) ---
        links = []
        search_area = content_container if content_container else soup # Search within container if found, else whole soup

        # Look for links specifically within list items often used for navigation
        link_elements = search_area.select("li a[href*='/grammar/british-grammar/'], div a[href*='/grammar/british-grammar/']")

        if not link_elements: # Fallback to broader search if specific links not found
             link_elements = search_area.select("a[href*='/grammar/british-grammar/']")

        added_urls = set() # To avoid duplicate links
        for a in link_elements:
            href = a.get("href", "")
            # Ensure it's a valid-looking grammar link and not the current page
            if href and "/grammar/british-grammar/" in href and href != url:
                link_url = href
                if href.startswith("/"):
                    link_url = "https://dictionary.cambridge.org" + href
                elif not href.startswith("http"):
                     # Handle relative URLs if necessary, though less common here
                     from urllib.parse import urljoin
                     link_url = urljoin(url, href)

                link_title = a.text.strip()
                if not link_title:
                     # Try to get title from URL segment
                     link_title = href.split('/')[-1].replace('-', ' ').title() if href.split('/')[-1] else "Related Grammar"

                # Check if URL already added
                if link_url not in added_urls:
                    links.append({
                        "title": link_title,
                        "url": link_url
                    })
                    added_urls.add(link_url) # Mark URL as added


        # --- 결과 생성 ---
        MIN_CONTENT_LENGTH = 150 # Increased minimum length for better quality
        if content and len(content) >= MIN_CONTENT_LENGTH:
            return {
                "id": f"cambridge_{title[:50].replace(' ', '_').replace('/', '_').lower()}", # More robust ID
                "source": "Cambridge Grammar",
                "title": title,
                "content": content,
                "url": url,
                "related_links": links[:15] # Allow slightly more links
            }
        else:
            print(f"Content too short or not found ({len(content)} chars) for: {url}")
            # Save skipped URL/content for debugging
            skipped_path = os.path.join(OUTPUT_DIR, f"__skipped_{page_name}.txt")
            with open(skipped_path, 'w', encoding='utf-8') as f:
                f.write(f"URL: {url}\nTitle: {title}\nContent Found ({len(content)} chars):\n{content}")
            return None

    except Exception as e:
        print(f"!!! Page crawling error: {url} - {e}")
        import traceback
        traceback.print_exc() # Print full traceback for errors
        return None

# --- [Grammar Categories List - Keep as is] ---
GRAMMAR_CATEGORIES = [
    "adjectives-and-adverbs", "clauses-and-sentences", "determiners-and-quantifiers",
    "ellipsis", "function-words", "idioms", "modality", "modifiers-in-nounal-groups",
    "negation", "nouns", "past", "phrasal-verbs", "pronouns", "questions",
    "relative-clauses", "reported-speech", "verbs",
    # Add more specific pages if known, e.g.:
    "adjectives", "adverbs", "nouns-countable-and-uncountable"
]

# --- [Main Execution Block - Revised] ---
if __name__ == "__main__":
    try:
        print("Cambridge 문법 콘텐츠 크롤링 시작 (Refined)...")
        all_docs = []
        crawled_urls = set() # Keep track of all attempted URLs

        # Function to process a URL if not already crawled
        def process_url(url_to_crawl, source_title="N/A"):
            if url_to_crawl in crawled_urls:
                print(f"  이미 크롤링됨: {url_to_crawl}")
                return None # Skip if already processed

            print(f"-> 크롤링 시도 ({source_title}): {url_to_crawl}")
            crawled_urls.add(url_to_crawl)
            page_doc = crawl_grammar_page(url_to_crawl)
            time.sleep(1.8) # Maintain delay

            if page_doc:
                print(f"✓ 성공: {page_doc['title']} (내용: {len(page_doc['content'])}자, 링크: {len(page_doc['related_links'])})")
                return page_doc
            else:
                print(f"✗ 실패 또는 내용 부족: {url_to_crawl}")
                return None

        # Start with the main grammar index page
        base_url = "https://dictionary.cambridge.org/grammar/british-grammar/"
        main_page_doc = process_url(base_url, "Main Page")

        urls_to_process = []
        if main_page_doc:
            all_docs.append(main_page_doc)
            urls_to_process.extend([link['url'] for link in main_page_doc.get('related_links', [])])

        # Add direct category URLs to the list
        for category in GRAMMAR_CATEGORIES:
            category_url = f"https://dictionary.cambridge.org/grammar/british-grammar/{category}"
            if category_url not in crawled_urls and category_url not in urls_to_process:
                 urls_to_process.append(category_url)

        # Process discovered URLs and category URLs
        processed_count = 0
        max_pages = 100 # Limit the total number of pages to avoid excessive crawling
        initial_url_count = len(urls_to_process)
        print(f"\n큐에 추가된 URL {initial_url_count}개 처리 시작 (최대 {max_pages}개)...")

        while urls_to_process and processed_count < max_pages:
            current_url = urls_to_process.pop(0) # Get the next URL from the front
            doc = process_url(current_url, "Queue")
            if doc:
                 all_docs.append(doc)
                 processed_count += 1
                 # Add newly found related links to the end of the queue (breadth-first approach)
                 new_links_added = 0
                 for link in doc.get('related_links', []):
                     new_url = link['url']
                     # Basic validation for URL format
                     if new_url.startswith("https://dictionary.cambridge.org/grammar/british-grammar/") and \
                        new_url not in crawled_urls and \
                        new_url not in urls_to_process:
                         urls_to_process.append(new_url)
                         new_links_added +=1
                 if new_links_added > 0:
                    print(f"  큐에 새 링크 {new_links_added}개 추가됨 (현재 큐 크기: {len(urls_to_process)})")


        # 중복 제거 (ID 기준 - ID 생성을 더 신뢰성있게 만들었으므로)
        unique_docs_map = {}
        for doc in all_docs:
            if doc['id'] not in unique_docs_map:
                 unique_docs_map[doc['id']] = doc

        unique_docs = list(unique_docs_map.values())

        print(f"\nCambridge 문법 콘텐츠 수집 완료: {len(unique_docs)}개 고유 문서 (총 {len(crawled_urls)}개 URL 시도)")

        # 결과 저장
        output_filename = "cambridge_grammar_documents_refined.json"
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(unique_docs, f, ensure_ascii=False, indent=2)

        print(f"문서 저장 완료 → {output_filename}")

    except Exception as e:
        print(f"!!! 메인 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 브라우저 종료
        if 'driver' in locals() and driver:
            driver.quit()
            print("WebDriver 종료됨.")