from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import os
import time
import datetime
import requests
from urllib.parse import quote
from config import FILE_PATHS, WEBSITE_URLS
from contextlib import contextmanager


def load_existing_data():
    if os.path.exists(FILE_PATHS.STOCK_LISTINGS):
        return pd.read_excel(FILE_PATHS.STOCK_LISTINGS)
    return pd.DataFrame({'Stock Name': [], 'URL': []})

def save_stock_counts(counts):
    count_df = pd.DataFrame(list(counts.items()), columns=['Letter', 'Count'])
    count_df = count_df.sort_values('Letter')
    count_df.to_excel(FILE_PATHS.STOCK_DATA, index=False)
    print("Stock counts saved to stockData.xlsx")

def scrape_stocks():
    existing_df = load_existing_data()
    stock_names = existing_df['Stock Name'].tolist()
    stock_links = existing_df['URL'].tolist()
    rescan_letters = ['d', 'e', 'i', 's', 'x']
    letter_counts = {}
    driver = webdriver.Chrome()
    base_url = WEBSITE_URLS.TICKERTAPE_STOCKS
    
    try:
        for letter in rescan_letters:
            print(f"Rescanning stocks starting with {letter.upper()}...")
            driver.get(base_url + letter)
            from time import sleep
            sleep(3)
            stock_elements = driver.find_elements(By.CSS_SELECTOR, "div.jsx-1528870203 a")
            letter_counts[letter.upper()] = len(stock_elements)
            
            for element in stock_elements:
                name = element.text
                link = element.get_attribute('href')
                if name and link and name not in stock_names:
                    stock_names.append(name)
                    stock_links.append(link)
            print(f"Found {len(stock_elements)} stocks for letter {letter.upper()}")

        other_counts = {
            'A': 504, 'B': 252, 'C': 251, 'F': 84, 'G': 268, 'H': 172,
            'J': 133, 'K': 237, 'L': 112, 'M': 371, 'N': 208, 'O': 93,
            'P': 296, 'Q': 21, 'R': 250, 'T': 233, 'U': 100, 'V': 188,
            'W': 68, 'Y': 27, 'Z': 35
        }
        letter_counts.update(other_counts)
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        
    finally:
        driver.quit()
        df = pd.DataFrame({
            'Stock Name': stock_names,
            'URL': stock_links
        })
        df.to_excel(FILE_PATHS.STOCK_LISTINGS, index=False)
        print(f"Data saved to {FILE_PATHS.STOCK_LISTINGS}")
        save_stock_counts(letter_counts)
        return df

def filter_stock_listings():
    if not os.path.exists(FILE_PATHS.STOCK_LISTINGS):
        print(f"File not found: {FILE_PATHS.STOCK_LISTINGS}")
        return
    
    df = pd.read_excel(FILE_PATHS.STOCK_LISTINGS)
    filtered_df = df[df['URL'].str.contains('stock', case=False, na=False)]
    filtered_df.to_excel(FILE_PATHS.FILTERED_STOCK_LISTINGS, index=False)
    print(f"Filtered {len(df)} entries to {len(filtered_df)} stocks")
    print(f"Saved filtered data to {FILE_PATHS.FILTERED_STOCK_LISTINGS}")
    return filtered_df

def get_letter_input():
    """Ask the user which letter to process."""
    letter = input("Enter the letter to process (A-Z): ").strip().upper()
    if len(letter) != 1 or not letter.isalpha():
        print("Please enter a single letter (A-Z)")
        return get_letter_input()
    return letter

def load_annual_report_data():
    """Load existing annual report data and ensure all required columns exist."""
    df = None
    if os.path.exists(FILE_PATHS.FILTERED_STOCK_LISTINGS):
        print(f"Loading existing data from {FILE_PATHS.FILTERED_STOCK_LISTINGS}")
        df = pd.read_excel(FILE_PATHS.FILTERED_STOCK_LISTINGS)
        print(f"Loaded {len(df)} existing records")
    else:
        print("No existing data found, creating new DataFrame")
        df = pd.DataFrame({'Stock Name': [], 'URL': []})
    
    # Ensure all required columns exist
    report_years = get_report_years()
    required_columns = ['Latest Report'] + [f"Report {year}" for year in report_years]
    
    # Add presentation columns
    presentation_columns = ['Latest Presentation'] + [f"Presentation {year}" for year in report_years]
    required_columns.extend(presentation_columns)
    
    for column in required_columns:
        if column not in df.columns:
            print(f"Adding missing column: {column}")
            df[column] = ""
    
    return df

def get_report_years():
    """Get the range of years for annual reports (current year to 5 years back)."""
    current_year = datetime.datetime.now().year
    latest_year = current_year if datetime.datetime.now().month > 6 else current_year - 1
    return list(range(latest_year, latest_year - 6, -1))

def search_screener_api(stock_name: str) -> str:
    """Search for a stock using Screener.in's search API."""
    try:
        # Format query string, removing any special characters
        query = quote(stock_name.lower().split()[0])  # Use first word of stock name
        search_url = f"https://www.screener.in/api/company/search/?q={query}&v=3&fts=1"
        
        print(f"Searching Screener.in API with query: {query}")
        response = requests.get(search_url)
        
        if response.status_code == 200:
            results = response.json()
            
            if not results:
                print("No results found in Screener.in API")
                return None
            
            # Filter out the "Search everywhere" result
            company_results = [r for r in results if r['id'] is not None]
            
            if not company_results:
                print("No company matches found")
                return None
            
            # Find best match using fuzzy matching
            from difflib import SequenceMatcher
            
            def similarity(a, b):
                return SequenceMatcher(None, a.lower(), b.lower()).ratio()
            
            # Sort results by name similarity
            company_results.sort(
                key=lambda x: similarity(x['name'], stock_name),
                reverse=True
            )
            
            best_match = company_results[0]
            match_score = similarity(best_match['name'], stock_name)
            
            if match_score > 0.6:  # Threshold for accepting a match
                print(f"Found match: {best_match['name']} (similarity: {match_score:.2f})")
                return f"https://www.screener.in{best_match['url']}"
            else:
                print(f"Best match {best_match['name']} below similarity threshold ({match_score:.2f})")
                return None
                
        else:
            print(f"API request failed with status code: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error searching Screener.in API: {e}")
        return None

@contextmanager
def save_debug_html(stock_name: str):
    """Temporarily save HTML for debugging and cleanup after use."""
    debug_file = f"debug_{stock_name.replace(' ', '_')}.html"
    try:
        yield debug_file
    finally:
        # Clean up the debug file if it exists
        if os.path.exists(debug_file):
            try:
                os.remove(debug_file)
                print(f"Cleaned up debug file: {debug_file}")
            except Exception as e:
                print(f"Warning: Could not delete debug file {debug_file}: {e}")

def extract_annual_reports_and_presentations(driver, stock_name: str):
    """Extract annual reports and presentation links from the page."""
    try:
        with save_debug_html(stock_name) as debug_file:
            page_source = driver.page_source
            with open(debug_file, "w", encoding="utf-8") as f:
                f.write(page_source)
            
            # Results to store
            reports = {}
            presentations = {}
            latest_report = ""
            latest_presentation = ""
            
            # 1. Extract Annual Reports
            annual_report_locators = [
                "div.documents.annual-reports", 
                "//h3[contains(text(), 'Annual reports')]/..",
                "//div[contains(@class, 'annual-reports')]"
            ]
            
            annual_reports_div = None
            for locator in annual_report_locators:
                try:
                    if locator.startswith("//"):
                        elements = driver.find_elements(By.XPATH, locator)
                    else:
                        elements = driver.find_elements(By.CSS_SELECTOR, locator)
                    
                    if elements:
                        annual_reports_div = elements[0]
                        print(f"Found annual reports section using selector: {locator}")
                        break
                except Exception:
                    continue
            
            # Process annual reports if found
            if annual_reports_div:
                try:
                    # Try the most specific selector first
                    links = annual_reports_div.find_elements(By.CSS_SELECTOR, "ul.list-links li a")
                    if not links:
                        # Try a more generic selector
                        links = annual_reports_div.find_elements(By.TAG_NAME, "a")
                    
                    print(f"Found {len(links)} potential annual report links")
                    
                    # Process each annual report link
                    for link in links:
                        try:
                            link_text = link.text.strip()
                            url = link.get_attribute('href')
                            
                            if not url or not link_text:
                                continue
                                
                            print(f"Processing annual report link: {link_text} -> {url}")
                            
                            # Match different formats: "Financial Year 2024", "Annual Report 2024", just "2024"
                            year = None
                            
                            if "Financial Year" in link_text:
                                year = link_text.replace("Financial Year", "").strip()
                            elif "Annual Report" in link_text:
                                year = link_text.replace("Annual Report", "").strip()
                            elif link_text.isdigit() and len(link_text) == 4:
                                year = link_text
                            else:
                                # Try to extract a 4-digit year from the text
                                import re
                                year_match = re.search(r'\b(20\d{2})\b', link_text)
                                if year_match:
                                    year = year_match.group(1)
                            
                            if year:
                                report_key = f"Report {year}"
                                reports[report_key] = url
                                
                                if not latest_report:
                                    latest_report = url
                        except Exception as e:
                            print(f"Error processing annual report link: {e}")
                    
                    print(f"Successfully extracted {len(reports)} annual reports")
                except Exception as e:
                    print(f"Error extracting annual report links: {e}")
            else:
                print("No annual reports section found")
            
            # 2. Extract Concall Presentations
            concall_locators = [
                "div.documents.concalls", 
                "//h3[contains(text(), 'Concalls')]/..",
                "//div[contains(@class, 'concalls')]"
            ]
            
            concall_div = None
            for locator in concall_locators:
                try:
                    if locator.startswith("//"):
                        elements = driver.find_elements(By.XPATH, locator)
                    else:
                        elements = driver.find_elements(By.CSS_SELECTOR, locator)
                    
                    if elements:
                        concall_div = elements[0]
                        print(f"Found concalls section using selector: {locator}")
                        break
                except Exception:
                    continue
            
            # Process concall presentations if found
            if concall_div:
                try:
                    # Find all list items containing presentation links
                    list_items = concall_div.find_elements(By.CSS_SELECTOR, "li.flex")
                    print(f"Found {len(list_items)} potential concall items")
                    
                    # Current year for reference
                    current_year = datetime.datetime.now().year
                    
                    # Process each list item to extract date and presentation link
                    for item in list_items:
                        try:
                            # Extract the date/period
                            date_div = item.find_element(By.CSS_SELECTOR, "div.ink-600.font-size-15")
                            date_text = date_div.text.strip()
                            print(f"Processing concall item: {date_text}")
                            
                            # Extract month and year
                            import re
                            month_year_match = re.match(r'(\w+)\s+(20\d{2})', date_text)
                            
                            if not month_year_match:
                                print(f"Could not parse date format: {date_text}")
                                continue
                                
                            month, year = month_year_match.groups()
                            year = int(year)
                            
                            # Find the PPT link in this item if it exists
                            ppt_links = [link for link in item.find_elements(By.TAG_NAME, "a") 
                                        if link.text.strip() == "PPT"]
                            
                            if ppt_links:
                                url = ppt_links[0].get_attribute('href')
                                print(f"Found presentation for {month} {year}: {url}")
                                
                                presentation_key = f"Presentation {year}"
                                
                                # Only add if not already present or if this is more recent for the same year
                                if (presentation_key not in presentations 
                                    or presentations.get(f"Month {year}") is None 
                                    or month_value(month) > month_value(presentations[f"Month {year}"])):
                                    
                                    presentations[presentation_key] = url
                                    presentations[f"Month {year}"] = month
                                    
                                    # Update latest presentation if this is the most recent
                                    if not latest_presentation or (year == current_year or 
                                                                (latest_presentation == "" and year == current_year - 1)):
                                        latest_presentation = url
                        except Exception as e:
                            print(f"Error processing concall item: {e}")
                    
                    # Clean up temporary month keys
                    for year in range(current_year - 5, current_year + 1):
                        if f"Month {year}" in presentations:
                            del presentations[f"Month {year}"]
                    
                    print(f"Successfully extracted {len(presentations)} presentation links")
                except Exception as e:
                    print(f"Error extracting presentation links: {e}")
            else:
                print("No concalls section found")
            
            # Combine results
            result = {}
            if latest_report:
                result["Latest Report"] = latest_report
            if latest_presentation:
                result["Latest Presentation"] = latest_presentation
                
            # Add all reports and presentations to the result
            result.update(reports)
            result.update(presentations)
            
            return result
            
    except Exception as e:
        print(f"Error extracting data: {e}")
        return {}

def month_value(month_name):
    """Convert month name to numeric value for comparison."""
    months = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    return months.get(month_name[:3], 0)  # Use first 3 chars to match month abbreviations

def search_and_scrape_annual_reports(stock_name, driver):
    """Search for stock on Google and scrape annual reports from Screener."""
    try:
        # First try the API search
        screener_url = search_screener_api(stock_name)
        if screener_url:
            print(f"Using Screener.in API URL: {screener_url}")
            driver.get(screener_url)
            time.sleep(3)
            
            # Extract annual reports and presentation links
            result = extract_annual_reports_and_presentations(driver, stock_name)
            return result
        else:
            print(f"Could not find stock {stock_name} using Screener.in API")
            return {}
        
    except Exception as e:
        print(f"Error searching for {stock_name}: {e}")
        return {}

def process_stocks_by_letter(letter):
    """Process all stocks starting with the specified letter."""
    print(f"\n{'='*80}\nProcessing stocks starting with letter {letter}\n{'='*80}")
    
    # Load data
    filtered_df = load_annual_report_data()
    print(f"Total records in database: {len(filtered_df)}")
    
    # Filter by letter
    letter_stocks = filtered_df[filtered_df['Stock Name'].str.startswith(letter, na=False)]
    print(f"Found {len(letter_stocks)} stocks starting with letter {letter}")
    
    if len(letter_stocks) == 0:
        print("No stocks found for this letter.")
        return
    
    # Configure Chrome options
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)
    
    driver = webdriver.Chrome(options=chrome_options)
    driver.execute_cdp_cmd('Network.setUserAgentOverride', {
        "userAgent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'
    })
    
    try:
        processed_count = 0
        skipped_report_count = 0
        skipped_presentation_count = 0
        success_report_count = 0
        success_presentation_count = 0
        error_count = 0
        
        for index, row in letter_stocks.iterrows():
            stock_name = row['Stock Name']
            print(f"\n{'-'*80}")
            print(f"Processing [{processed_count + 1}/{len(letter_stocks)}] {stock_name}")
            
            # Check what data is already available
            has_report_data = pd.notna(row.get('Latest Report', '')) and row['Latest Report'] != ''
            has_presentation_data = pd.notna(row.get('Latest Presentation', '')) and row['Latest Presentation'] != ''
            
            if has_report_data and has_presentation_data:
                print(f"Stock {stock_name} already has both annual report and presentation URLs. Skipping...")
                skipped_report_count += 1
                skipped_presentation_count += 1
                processed_count += 1
                continue
            
            try:
                # Scrape new data
                print(f"Searching for data for {stock_name}...")
                new_data = search_and_scrape_annual_reports(stock_name, driver)
                
                # Update dataframe with new data
                if new_data:
                    reports_found = False
                    presentations_found = False
                    
                    # Update each column if we have new data and it's not already populated
                    for col, value in new_data.items():
                        if col not in filtered_df.columns:
                            print(f"Warning: Column '{col}' not found in dataframe")
                            continue
                            
                        # For annual reports
                        if col.startswith("Report") or col == "Latest Report":
                            if not has_report_data or (pd.isna(filtered_df.loc[index, col]) or filtered_df.loc[index, col] == ""):
                                filtered_df.loc[index, col] = value
                                reports_found = True
                        
                        # For presentations
                        if col.startswith("Presentation") or col == "Latest Presentation":
                            if not has_presentation_data or (pd.isna(filtered_df.loc[index, col]) or filtered_df.loc[index, col] == ""):
                                filtered_df.loc[index, col] = value
                                presentations_found = True
                    
                    if reports_found:
                        success_report_count += 1
                        print(f"Added annual report URLs for {stock_name}")
                    elif has_report_data:
                        skipped_report_count += 1
                        print(f"Kept existing annual report URLs for {stock_name}")
                    
                    if presentations_found:
                        success_presentation_count += 1
                        print(f"Added presentation URLs for {stock_name}")
                    elif has_presentation_data:
                        skipped_presentation_count += 1
                        print(f"Kept existing presentation URLs for {stock_name}")
                        
                    if not reports_found and not presentations_found:
                        print(f"No new data found for {stock_name}")
                        error_count += 1
                else:
                    print(f"No data found for {stock_name}")
                    error_count += 1
                
                # Save progress
                filtered_df.to_excel(FILE_PATHS.FILTERED_STOCK_LISTINGS, index=False)
                print(f"Progress saved - Reports: {success_report_count} added, {skipped_report_count} skipped; "
                      f"Presentations: {success_presentation_count} added, {skipped_presentation_count} skipped; "
                      f"Errors: {error_count}")
                
                processed_count += 1
                time.sleep(1)
                
            except Exception as e:
                print(f"Error processing {stock_name}: {str(e)}")
                error_count += 1
                continue
    
    except Exception as e:
        print(f"An error occurred during processing: {str(e)}")
    
    finally:
        driver.quit()
        filtered_df.to_excel(FILE_PATHS.FILTERED_STOCK_LISTINGS, index=False)
        
        # Print summary
        print(f"\n{'='*80}")
        print("Processing Summary:")
        print(f"Total stocks for letter {letter}: {len(letter_stocks)}")
        print(f"Annual Reports: {success_report_count} added, {skipped_report_count} skipped")
        print(f"Presentations: {success_presentation_count} added, {skipped_presentation_count} skipped")
        print(f"Failed to process: {error_count}")
        print(f"Data saved to: {FILE_PATHS.FILTERED_STOCK_LISTINGS}")
        print('='*80)

def analyze_filtered_stock_listings():
    """Analyze the filtered stock listings file and display comprehensive information."""
    if not os.path.exists(FILE_PATHS.FILTERED_STOCK_LISTINGS):
        print(f"Error: File not found: {FILE_PATHS.FILTERED_STOCK_LISTINGS}")
        return
    
    print(f"\n{'='*80}\nAnalyzing Filtered Stock Listings\n{'='*80}")
    
    try:
        # Load the data
        df = pd.read_excel(FILE_PATHS.FILTERED_STOCK_LISTINGS)
        
        # Basic information
        print(f"\n1. Basic Information:")
        print(f"   - File location: {FILE_PATHS.FILTERED_STOCK_LISTINGS}")
        print(f"   - Total records: {len(df)}")
        print(f"   - Total columns: {len(df.columns)}")
        print(f"   - File size: {os.path.getsize(FILE_PATHS.FILTERED_STOCK_LISTINGS) / (1024*1024):.2f} MB")
        
        # Column information
        print(f"\n2. Column Information:")
        print(f"   - All columns ({len(df.columns)}):")
        for i, col in enumerate(df.columns):
            print(f"     {i+1}. {col}")
        
        # Data types
        print(f"\n3. Data Types:")
        for col, dtype in df.dtypes.items():
            print(f"   - {col}: {dtype}")
        
        # Missing values analysis
        print(f"\n4. Missing Values Analysis:")
        missing_values = df.isna().sum()
        for col, count in missing_values.items():
            percentage = (count / len(df)) * 100
            print(f"   - {col}: {count} missing ({percentage:.2f}%)")
        
        # Stock name analysis by first letter
        print(f"\n5. Stock Count by First Letter:")
        letter_counts = {}
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            count = len(df[df['Stock Name'].str.startswith(letter, na=False)])
            if count > 0:
                letter_counts[letter] = count
        
        for letter, count in sorted(letter_counts.items()):
            print(f"   - {letter}: {count} stocks")
        
        # Annual reports availability
        print(f"\n6. Annual Reports Availability:")
        report_years = [int(col.replace("Report ", "")) for col in df.columns if col.startswith("Report 20")]
        report_years.sort(reverse=True)
        
        for year in report_years:
            col = f"Report {year}"
            available = (~df[col].isna() & (df[col] != "")).sum()
            percentage = (available / len(df)) * 100
            print(f"   - {year}: {available} stocks ({percentage:.2f}%)")
        
        # Presentation availability
        print(f"\n7. Presentations Availability:")
        pres_years = [int(col.replace("Presentation ", "")) for col in df.columns if col.startswith("Presentation 20")]
        pres_years.sort(reverse=True)
        
        for year in pres_years:
            col = f"Presentation {year}"
            available = (~df[col].isna() & (df[col] != "")).sum()
            percentage = (available / len(df)) * 100
            print(f"   - {year}: {available} stocks ({percentage:.2f}%)")
        
        # Latest reports and presentations
        print(f"\n8. Latest Information:")
        latest_report_count = (~df['Latest Report'].isna() & (df['Latest Report'] != "")).sum()
        latest_pres_count = (~df['Latest Presentation'].isna() & (df['Latest Presentation'] != "")).sum()
        
        print(f"   - Stocks with latest report: {latest_report_count} ({latest_report_count / len(df) * 100:.2f}%)")
        print(f"   - Stocks with latest presentation: {latest_pres_count} ({latest_pres_count / len(df) * 100:.2f}%)")
        
        # Completeness analysis - stocks with all years of data
        print(f"\n9. Data Completeness Analysis:")
        
        # For annual reports
        all_report_cols = [f"Report {year}" for year in report_years]
        stocks_with_all_reports = df[all_report_cols].notna().all(axis=1).sum()
        print(f"   - Stocks with all years of annual reports: {stocks_with_all_reports} ({stocks_with_all_reports / len(df) * 100:.2f}%)")
        
        # For presentations
        all_pres_cols = [f"Presentation {year}" for year in pres_years]
        stocks_with_all_pres = df[all_pres_cols].notna().all(axis=1).sum()
        print(f"   - Stocks with all years of presentations: {stocks_with_all_pres} ({stocks_with_all_pres / len(df) * 100:.2f}%)")
        
        # Domain analysis for reports and presentations
        print(f"\n10. Link Domain Analysis:")
        
        # Sample some domains from reports
        report_domains = {}
        for col in ['Latest Report'] + all_report_cols:
            domains = df[col].dropna().apply(lambda x: extract_domain(x) if isinstance(x, str) and x else None).dropna()
            for domain, count in domains.value_counts().head(3).items():
                report_domains[domain] = report_domains.get(domain, 0) + count
        
        print("   - Top report domains:")
        for domain, count in sorted(report_domains.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"     * {domain}: {count} links")
        
        # Sample some domains from presentations
        pres_domains = {}
        for col in ['Latest Presentation'] + all_pres_cols:
            domains = df[col].dropna().apply(lambda x: extract_domain(x) if isinstance(x, str) and x else None).dropna()
            for domain, count in domains.value_counts().head(3).items():
                pres_domains[domain] = pres_domains.get(domain, 0) + count
        
        print("   - Top presentation domains:")
        for domain, count in sorted(pres_domains.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"     * {domain}: {count} links")
        
        print(f"\n{'='*80}")
        
    except Exception as e:
        print(f"Error analyzing file: {e}")

def extract_domain(url):
    """Extract domain from a URL."""
    try:
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        return domain
    except:
        return None

def process_single_stock(stock_ticker: str):
    """Process a single stock by its ticker name to fetch annual reports and presentations."""
    print(f"\n{'='*80}\nProcessing stock: {stock_ticker}\n{'='*80}")
    
    # Load existing data
    filtered_df = load_annual_report_data()
    
    # Setup Chrome
    driver = None
    try:
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option("useAutomationExtension", False)
        
        driver = webdriver.Chrome(options=chrome_options)
        driver.execute_cdp_cmd('Network.setUserAgentOverride', {
            "userAgent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'
        })
        
        # Search for the stock on Screener.in and scrape data
        screener_url = search_screener_api(stock_ticker)
        if screener_url:
            new_data = search_and_scrape_annual_reports(stock_ticker, driver)
            if new_data:
                # Create new row
                new_row = pd.DataFrame({
                    'Stock Name': [stock_ticker],
                    'URL': [screener_url]
                })
                
                # Add the annual report and presentation data
                for col, value in new_data.items():
                    new_row[col] = value
                
                # Append to existing DataFrame
                filtered_df = pd.concat([filtered_df, new_row], ignore_index=True)
                print("Successfully added new stock data")
                
                # Save changes
                filtered_df.to_excel(FILE_PATHS.FILTERED_STOCK_LISTINGS, index=False)
                print(f"Data saved to {FILE_PATHS.FILTERED_STOCK_LISTINGS}")
            else:
                print("Could not find annual reports or presentations")
        else:
            print("Could not find stock on Screener.in")
            
    except Exception as e:
        print(f"Error processing stock {stock_ticker}: {str(e)}")
    finally:
        if driver:
            driver.quit()

if __name__ == "__main__":
    action = input("Select action (1: Scrape stock data, 2: Filter stock listings, 3: Process stocks by letter, "
                  "4: Analyze filtered stock listings, 5: Process single stock): ")
    if action == '1':
        print("Scraping stock data from Ticker Tape...")
        scrape_stocks()
    elif action == '2':
        print("Filtering stock listings to contain only stock data...")
        filter_stock_listings()
    elif action == '3':
        letter = get_letter_input()
        print(f"Processing stocks starting with {letter} to append annual report URLs...")
        process_stocks_by_letter(letter)
    elif action == '4':
        print("Analyzing filtered stock listings...")
        analyze_filtered_stock_listings()
    elif action == '5':
        stock_ticker = input("Enter stock ticker name: ")
        process_single_stock(stock_ticker)
    else:
        print("Invalid action selected")
        print("Available options:")
        print("1: Scrape stock data from Ticker Tape")
        print("2: Filter stock listings to contain only stock data")
        print("3: Process stocks by letter to append annual report URLs")
        print("4: Analyze filtered stock listings data")
        print("5: Process single stock by ticker name")
