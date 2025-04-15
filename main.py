from pathlib import Path
from stock_search import StockSearch
import os
from dotenv import load_dotenv
from config import REQUIRED_ENV_VARS, DATA_DIR
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from background_tasks import download_stock_documents, check_existing_downloads
from utils import find_stock_in_listings 
from financial_analysis import generate_financial_analysis, generate_dcf_analysis

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000", "https://financial-first.vercel.app"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Origin", "Accept"],
        "expose_headers": ["Content-Type", "Authorization"]
    }
})

def check_env_vars():
    """Check if all required environment variables are set."""
    load_dotenv()
    missing_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these variables in your .env file.")
        return False
    return True

@app.route('/api/search', methods=['GET'])
def search_stocks():
    try:
        query = request.args.get('q', '')
        if not query or len(query) < 3:
            return jsonify([]), 200  # Return empty array instead of 404
        
        output_dir = DATA_DIR
        output_dir.mkdir(exist_ok=True)
        
        searcher = StockSearch(output_dir)
        results = searcher.instant_search(query)
        
        return jsonify(results), 200  # Always return 200 with results (even empty)
    except Exception as e:
        print(f"Search error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/stock/save', methods=['POST'])
def save_stock():
    stock_data = request.json
    if not stock_data:
        return jsonify({"success": False, "error": "No stock data provided"}), 400
    
    output_dir = DATA_DIR
    output_dir.mkdir(exist_ok=True)
    
    # Get ticker symbol from the stock data
    ticker = stock_data.get('ticker', '') or stock_data.get('symbol', '')
    
    if not ticker:
        return jsonify({"success": False, "error": "No ticker symbol provided in stock data"}), 400
    
    # Save the stock data
    searcher = StockSearch(output_dir)
    success = searcher.save_stock_data(stock_data)
    
    if not success:
        return jsonify({"success": False, "error": "Failed to save stock data"}), 500
    
    # Check if documents are already downloaded
    ticker_dir = DATA_DIR / ticker.lower()
    if check_existing_downloads(ticker_dir):
        return jsonify({
            "success": True,
            "message": "Stock data saved. Documents already downloaded."
        })
    
    # Start background task to download reports and presentations
    try:
        stock_info = find_stock_in_listings(ticker)
        download_stock_documents(ticker, stock_info)
        
        return jsonify({
            "success": True,
            "message": "Stock data saved. Reports and presentations will be downloaded in the background."
        })
    except Exception as e:
        print(f"Error starting background task: {str(e)}")
        return jsonify({
            "success": True,
            "warning": "Stock data saved but background download failed to start."
        })

@app.route('/api/stocks/<symbol>', methods=['GET'])
def get_stock_data(symbol):
    try:
        # Convert symbol to lowercase as the folder structure uses lowercase
        symbol_lower = symbol.lower()
        stock_folder = DATA_DIR / symbol_lower
        stock_data_file = stock_folder / 'sData.json'
        
        # Check if the file exists
        if not stock_data_file.exists():
            return jsonify({"success": False, "error": f"Stock data for {symbol} not found"}), 404
        
        # Read the stock data from the JSON file
        with open(stock_data_file, 'r', encoding='utf-8') as f:
            stock_data = json.load(f)
        
        # Try to load balance sheet data if available
        balance_sheet_file = stock_folder / 'balancesheet.json'
        balance_growth_file = stock_folder / 'balGrowth.json'
        quarterly_file = stock_folder / 'quarterly.json'
        annual_file = stock_folder / 'annual.json'
        
        if balance_sheet_file.exists():
            with open(balance_sheet_file, 'r', encoding='utf-8') as f:
                balance_sheet_data = json.load(f)
                stock_data['balance_sheet'] = balance_sheet_data
        
        if balance_growth_file.exists():
            with open(balance_growth_file, 'r', encoding='utf-8') as f:
                balance_growth_data = json.load(f)
                stock_data['balance_growth'] = balance_growth_data
                
        if quarterly_file.exists():
            with open(quarterly_file, 'r', encoding='utf-8') as f:
                quarterly_data = json.load(f)
                stock_data['quarterly'] = quarterly_data
                
        if annual_file.exists():
            with open(annual_file, 'r', encoding='utf-8') as f:
                annual_data = json.load(f)
                stock_data['annual'] = annual_data
            
        return jsonify(stock_data)
        
    except Exception as e:
        print(f"Error fetching stock data for {symbol}: {str(e)}")
        return jsonify({"success": False, "error": f"Failed to fetch stock data for {symbol}"}), 500

@app.route('/api/stocks/<symbol>/analysis', methods=['GET'])
def get_stock_analysis(symbol):
    try:
        symbol_lower = symbol.lower()
        stock_folder = DATA_DIR / symbol_lower
        mapping_file = Path(__file__).parent / 'mapping.json'
        report_file = stock_folder / 'report.md'
        
        if not stock_folder.exists():
            return jsonify({"success": False, "error": f"Stock data for {symbol} not found"}), 404
        
        # Check if report already exists
        if report_file.exists():
            with open(report_file, 'r', encoding='utf-8') as f:
                report_content = f.read()
                # Clean any potential markdown code blocks
                if report_content.strip().startswith("```markdown"):
                    report_content = report_content.split("```markdown", 1)[1]
                    if "```" in report_content:
                        report_content = report_content.rsplit("```", 1)[0]
                return jsonify({"success": True, "analysis": report_content.strip()})
                
        # Generate new report if doesn't exist
        analysis = generate_financial_analysis(stock_folder, mapping_file)
        # Ensure we're not wrapping the analysis in code blocks before returning
        if analysis.strip().startswith("```markdown"):
            analysis = analysis.split("```markdown", 1)[1]
            if "```" in analysis:
                analysis = analysis.rsplit("```", 1)[0]
        return jsonify({"success": True, "analysis": analysis.strip()})
        
    except Exception as e:
        print(f"Error generating analysis for {symbol}: {str(e)}")
        return jsonify({"success": False, "error": f"Failed to generate analysis for {symbol}"}), 500

@app.route('/api/stocks/<symbol>/dcf', methods=['GET'])
def get_stock_dcf_analysis(symbol):
    try:
        symbol_lower = symbol.lower()
        stock_folder = DATA_DIR / symbol_lower
        report_file = stock_folder / 'dcf_report.md'
        
        if not stock_folder.exists():
            return jsonify({"success": False, "error": f"Stock data for {symbol} not found"}), 404
        
        # Check if DCF report already exists
        if report_file.exists():
            with open(report_file, 'r', encoding='utf-8') as f:
                report_content = f.read()
                if report_content.strip().startswith("```markdown"):
                    report_content = report_content.split("```markdown", 1)[1]
                    if "```" in report_content:
                        report_content = report_content.rsplit("```", 1)[0]
                return jsonify({"success": True, "analysis": report_content.strip()})
                
        # Generate new DCF report
        analysis = generate_dcf_analysis(stock_folder)
        if analysis.strip().startswith("```markdown"):
            analysis = analysis.split("```markdown", 1)[1]
            if "```" in analysis:
                analysis = analysis.rsplit("```", 1)[0]
        return jsonify({"success": True, "analysis": analysis.strip()})
        
    except Exception as e:
        print(f"Error generating DCF analysis for {symbol}: {str(e)}")
        return jsonify({"success": False, "error": f"Failed to generate DCF analysis for {symbol}"}), 500

def main():
    # Check environment variables
    if not check_env_vars():
        print("Environment check failed. Exiting...")
        return
    
    # Start the Flask app
    app.run(host='0.0.0.0', port=5000)

if __name__ == "__main__":
    main()

# This is required for Vercel serverless deployment
app.debug = False
