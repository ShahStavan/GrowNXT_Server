# Stock Fundamental Analysis Server

## Project Structure

```
server/
├── api/                    # API layer
│   ├── __init__.py
│   ├── app.py             # Flask application & routes
│   └── search.py          # Stock search functionality
├── services/              # Business logic layer
│   ├── __init__.py
│   ├── data_service.py    # Financial data fetching
│   ├── enrichment_service.py  # LLM-based enrichment
│   └── analysis_service.py    # Report generation
├── core/                  # Core utilities & config
│   ├── __init__.py
│   ├── config.py          # Configuration & constants
│   ├── utils.py           # Utility functions
│   ├── llm_config.py      # LLM client setup
│   └── prompts.py         # AI prompts
├── scripts/               # Standalone scripts
│   ├── __init__.py
│   ├── fetch_nifty50.py   # Fetch NIFTY 50 data
│   └── ticker_scraper.py  # Web scraping utilities
├── data/                  # Data storage
├── .env                   # Environment variables
├── requirements.txt       # Dependencies
└── run.py                 # Application entry point

```

## Running the Server

```bash
# Start the Flask server
python run.py

# Or directly
python -m api.app
```

## Running Scripts

```bash
# Fetch NIFTY 50 data
python -m scripts.fetch_nifty50

# Or using the old method
python scripts/fetch_nifty50.py
```

## Architecture

- **api/**: REST API endpoints using Flask
- **services/**: Business logic separated from API layer
- **core/**: Shared utilities, configuration, and constants
- **scripts/**: CLI tools and batch processing scripts
