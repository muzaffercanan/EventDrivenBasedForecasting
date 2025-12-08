"""
Web scraping script for SP500 news headlines

Scrapes news from multiple financial news sources to increase data volume.
Target: 10+ news per day

Sources:
- Yahoo Finance API (yfinance) - Historical news support
- Yahoo Finance SP500 news (web scraping - limited)
- MarketWatch SP500 news (web scraping - limited)
"""

import argparse
import os
import time
import pandas as pd
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')

# Try to import yfinance for Yahoo Finance API
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("Warning: yfinance not installed. Install with: pip install yfinance")

# User agent to avoid blocking
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}


def scrape_yahoo_finance_api(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Fetch SP500 news from Yahoo Finance using yfinance API
    
    This method can access historical news data (limited by Yahoo Finance API).
    """
    if not YFINANCE_AVAILABLE:
        print("  yfinance not available, skipping API method")
        return pd.DataFrame()
    
    print("Fetching Yahoo Finance SP500 news via API...")
    news_list = []
    
    try:
        # Get SP500 ticker
        ticker = yf.Ticker("^GSPC")
        
        # Fetch news (yfinance provides recent news, historical may be limited)
        news = ticker.news
        
        if news:
            for article in news:
                try:
                    title = article.get('title', '')
                    # Convert timestamp to date
                    pub_time = article.get('providerPublishTime', None)
                    if pub_time:
                        article_date = pd.to_datetime(pub_time, unit='s')
                    else:
                        article_date = datetime.now()
                    
                    # Filter by date range
                    if start_date <= article_date <= end_date:
                        if title and len(title) > 10:
                            news_list.append({
                                'Title': title,
                                'Date': article_date.strftime('%Y-%m-%d'),
                                'Source': 'Yahoo Finance API'
                            })
                except Exception as e:
                    continue
        
        time.sleep(1)  # Rate limiting
        
    except Exception as e:
        print(f"  Error fetching Yahoo Finance API: {e}")
    
    print(f"  Found {len(news_list)} articles from Yahoo Finance API")
    return pd.DataFrame(news_list)


def scrape_yahoo_finance_sp500(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Scrape SP500 news from Yahoo Finance (web scraping - fallback)
    
    Note: Yahoo Finance has rate limits and may require API access for historical data.
    This is a template - actual implementation may need adjustments.
    """
    print("Scraping Yahoo Finance SP500 news (web scraping)...")
    news_list = []
    
    # Yahoo Finance SP500 news URL
    base_url = "https://finance.yahoo.com/quote/%5EGSPC/news"
    
    try:
        response = requests.get(base_url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find news articles (structure may vary)
            articles = soup.find_all('div', class_='js-stream-content') or soup.find_all('li', class_='js-stream-content')
            
            for article in articles[:20]:  # Limit to recent articles
                try:
                    title_elem = article.find('h3') or article.find('a')
                    if title_elem:
                        title = title_elem.get_text(strip=True)
                        
                        # Try to find date
                        date_elem = article.find('time') or article.find('span', class_='date')
                        if date_elem:
                            date_str = date_elem.get('datetime') or date_elem.get_text(strip=True)
                            try:
                                article_date = pd.to_datetime(date_str)
                            except:
                                article_date = datetime.now()
                        else:
                            article_date = datetime.now()
                        
                        if title and len(title) > 10:
                            news_list.append({
                                'Title': title,
                                'Date': article_date.strftime('%Y-%m-%d'),
                                'Source': 'Yahoo Finance'
                            })
                except Exception as e:
                    continue
        
        time.sleep(2)  # Rate limiting
        
    except Exception as e:
        print(f"Error scraping Yahoo Finance: {e}")
    
    print(f"  Found {len(news_list)} articles from Yahoo Finance")
    return pd.DataFrame(news_list)


def scrape_marketwatch_sp500(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Scrape SP500 news from MarketWatch
    
    Note: MarketWatch may have anti-scraping measures. This is a template.
    """
    print("Scraping MarketWatch SP500 news...")
    news_list = []
    
    base_url = "https://www.marketwatch.com/investing/index/spx"
    
    try:
        response = requests.get(base_url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find news articles
            articles = soup.find_all('div', class_='article__content') or soup.find_all('h3', class_='article__headline')
            
            for article in articles[:20]:
                try:
                    title_elem = article.find('a') or article
                    if title_elem:
                        title = title_elem.get_text(strip=True)
                        
                        if title and len(title) > 10:
                            news_list.append({
                                'Title': title,
                                'Date': datetime.now().strftime('%Y-%m-%d'),
                                'Source': 'MarketWatch'
                            })
                except:
                    continue
        
        time.sleep(2)
        
    except Exception as e:
        print(f"Error scraping MarketWatch: {e}")
    
    print(f"  Found {len(news_list)} articles from MarketWatch")
    return pd.DataFrame(news_list)


def filter_sp500_relevant(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to keep only SP500/financial relevant news"""
    if df.empty:
        return df
    
    sp500_keywords = [
        'sp500', 's&p', 's&p 500', 'sp 500',
        'stock', 'market', 'financial', 'trading',
        'dow', 'nasdaq', 'index', 'economy',
        'fed', 'federal reserve', 'inflation',
        'earnings', 'revenue', 'profit'
    ]
    
    df['title_lower'] = df['Title'].str.lower()
    df['is_relevant'] = df['title_lower'].str.contains('|'.join(sp500_keywords), 
                                                         case=False, na=False, regex=True)
    
    filtered = df[df['is_relevant']].copy()
    filtered = filtered.drop(columns=['title_lower', 'is_relevant'])
    
    return filtered


def merge_with_existing(new_df: pd.DataFrame, existing_csv: str) -> pd.DataFrame:
    """Merge new scraped data with existing dataset"""
    if os.path.exists(existing_csv):
        existing_df = pd.read_csv(existing_csv)
        
        # Combine
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        
        # Remove duplicates based on title
        combined = combined.drop_duplicates(subset=['Title'], keep='first')
        
        # Sort by date
        combined['Date'] = pd.to_datetime(combined['Date'])
        combined = combined.sort_values('Date').reset_index(drop=True)
        
        return combined
    else:
        return new_df


def main():
    parser = argparse.ArgumentParser(description="Scrape SP500 news from multiple sources")
    parser.add_argument("--output", type=str, 
                       default="SP500_news/raw/sp500_headlines_scraped.csv",
                       help="Output CSV file")
    parser.add_argument("--existing", type=str,
                       default="SP500_news/raw/sp500_headlines_2008_2024.csv",
                       help="Existing CSV to merge with")
    parser.add_argument("--days_back", type=int, default=None,
                       help="Number of days back to scrape (if not specified, uses existing data range)")
    parser.add_argument("--start_date", type=str, default=None,
                       help="Start date (YYYY-MM-DD). If not specified, uses existing data range")
    parser.add_argument("--end_date", type=str, default=None,
                       help="End date (YYYY-MM-DD). Default: today")
    parser.add_argument("--sources", type=str, nargs="+",
                       default=["yahoo", "marketwatch"],
                       choices=["yahoo", "marketwatch"],
                       help="News sources to scrape")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Determine date range
    if args.start_date and args.end_date:
        # Use provided dates
        start_date = pd.to_datetime(args.start_date)
        end_date = pd.to_datetime(args.end_date)
    elif args.start_date:
        # Start date provided, end is today
        start_date = pd.to_datetime(args.start_date)
        end_date = datetime.now()
    elif args.days_back:
        # Days back provided
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days_back)
    else:
        # Use existing data range (2008-01-02 to 2024-03-04)
        if os.path.exists(args.existing):
            existing_df = pd.read_csv(args.existing)
            existing_df['Date'] = pd.to_datetime(existing_df['Date'])
            start_date = existing_df['Date'].min()
            end_date = existing_df['Date'].max()
            print(f"Using existing data range: {start_date.date()} to {end_date.date()}")
        else:
            # Default: last 30 days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
    
    print("=" * 60)
    print("SP500 NEWS SCRAPING")
    print("=" * 60)
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print(f"Total days: {(end_date - start_date).days}")
    print(f"Sources: {', '.join(args.sources)}")
    print()
    print("NOT: Web scraping gerçek zamanlı veri çekmek için tasarlandı.")
    print("Geçmiş veriler (2008-2024) için API veya RSS feed kullanılması önerilir.")
    print("Bu script sadece son haberleri çekebilir (bugünden geriye doğru).")
    print()
    
    all_news = []
    
    # Scrape from each source
    if "yahoo" in args.sources:
        # Try API first (better for historical data)
        yahoo_api_df = scrape_yahoo_finance_api(start_date, end_date)
        if not yahoo_api_df.empty:
            all_news.append(yahoo_api_df)
        
        # Fallback to web scraping
        yahoo_df = scrape_yahoo_finance_sp500(start_date, end_date)
        if not yahoo_df.empty:
            all_news.append(yahoo_df)
    
    if "marketwatch" in args.sources:
        mw_df = scrape_marketwatch_sp500(start_date, end_date)
        if not mw_df.empty:
            all_news.append(mw_df)
    
    if not all_news:
        print("\nNo news scraped. Check your internet connection and source availability.")
        return
    
    # Combine all sources
    combined_df = pd.concat(all_news, ignore_index=True)
    print(f"\nTotal scraped: {len(combined_df)} articles")
    
    # Filter for SP500 relevance
    filtered_df = filter_sp500_relevant(combined_df)
    print(f"After filtering: {len(filtered_df)} relevant articles")
    
    # Merge with existing data if provided
    if os.path.exists(args.existing):
        final_df = merge_with_existing(filtered_df, args.existing)
        print(f"After merging with existing: {len(final_df)} total articles")
    else:
        final_df = filtered_df
        print(f"Saved {len(final_df)} articles (no existing file to merge)")
    
    # Add CP column if missing (for compatibility)
    if 'CP' not in final_df.columns:
        final_df['CP'] = None  # Will need to be filled with actual price data
    
    # Save
    final_df.to_csv(args.output, index=False)
    print(f"\nSaved to: {args.output}")
    
    # Summary by date
    if not final_df.empty:
        final_df['Date'] = pd.to_datetime(final_df['Date'])
        daily_counts = final_df.groupby(final_df['Date'].dt.date).size()
        print(f"\nDaily news count (last 10 days):")
        print(daily_counts.tail(10))


if __name__ == "__main__":
    main()

