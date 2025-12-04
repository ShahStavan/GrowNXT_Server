from pathlib import Path
from stock_search import StockSearch
import os
from dotenv import load_dotenv
from config import REQUIRED_ENV_VARS, DATA_DIR
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from background_tasks import check_existing_downloads
from financial_analysis import generate_financial_analysis, generate_dcf_analysis
import requests
import logging

logging.basicConfig(level=logging.INFO)

BSE_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
    'Referer': 'https://www.bseindia.com/',
    'Upgrade-Insecure-Requests': '1',
}

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

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
    query = request.args.get('q', '')
    if not query or len(query) < 3:
        return jsonify([])
    
    output_dir = DATA_DIR
    output_dir.mkdir(exist_ok=True)
    
    searcher = StockSearch(output_dir)
    results = searcher.instant_search(query)
    
    return jsonify(results)

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
    
    # Return success response even if documents don't exist
    return jsonify({
        "success": True,
        "message": "Stock data saved successfully."
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

@app.route('/api/stocks/<symbol>/upload', methods=['POST'])
def upload_stock_files(symbol):
    try:
        symbol_lower = symbol.lower()
        stock_folder = DATA_DIR / symbol_lower
        stock_folder.mkdir(exist_ok=True)
        mapping_file = Path(__file__).parent / 'mapping.json'
        uploaded_files = []

        # Handle URL-based uploads
        if 'annual_url' in request.form:
            url = request.form['annual_url'].strip()
            if url:
                logging.info(f"Downloading annual report from URL: {url}")
                try:
                    # Use appropriate headers based on the URL
                    headers = BSE_HEADERS if 'bseindia.com' in url else None
                    response = requests.get(url, stream=True, timeout=30, headers=headers)
                    
                    if response.ok:
                        annual_path = stock_folder / 'annual_report.pdf'
                        total_size = 0
                        with open(annual_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                                    total_size += len(chunk)
                        
                        if total_size > 0:
                            logging.info(f"Successfully downloaded annual report to {annual_path} (Size: {total_size/1024:.1f}KB)")
                            uploaded_files.append('annual_report.pdf (from URL)')
                        else:
                            logging.error("Downloaded file is empty")
                            return jsonify({
                                "success": False,
                                "error": "Downloaded file is empty"
                            }), 400
                    else:
                        error_msg = f"Failed to download PDF. Status code: {response.status_code}"
                        if response.status_code == 403:
                            error_msg += " (Access Forbidden - Website may require authentication)"
                        logging.error(error_msg)
                        return jsonify({
                            "success": False,
                            "error": error_msg
                        }), 400
                except Exception as download_error:
                    logging.error(f"Error downloading PDF: {str(download_error)}")
                    return jsonify({
                        "success": False,
                        "error": f"Error downloading PDF: {str(download_error)}"
                    }), 500

        # Handle direct file uploads
        if 'annual' in request.files:
            annual = request.files['annual']
            annual_path = stock_folder / 'annual_report.pdf'
            annual.save(str(annual_path))
            uploaded_files.append('annual_report.pdf')
            
        if 'presentation' in request.files:
            presentation = request.files['presentation']
            presentation_path = stock_folder / 'presentation.pdf'
            presentation.save(str(presentation_path))
            uploaded_files.append('presentation.pdf')
        
        if 'presentation_url' in request.form and request.form['presentation_url'].strip():
            url = request.form['presentation_url'].strip()
            try:
                headers = BSE_HEADERS if 'bseindia.com' in url else None
                response = requests.get(url, stream=True, timeout=30, headers=headers)
                if response.ok:
                    presentation_path = stock_folder / 'presentation.pdf'
                    with open(presentation_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    uploaded_files.append('presentation.pdf (from URL)')
            except Exception as e:
                logging.error(f"Error downloading presentation from URL: {e}")

        # Regenerate analysis only once if files were uploaded
        if uploaded_files:
            try:
                logging.info("Regenerating financial analysis...")
                generate_financial_analysis(stock_folder, mapping_file)
                logging.info("Financial analysis regenerated successfully")
            except Exception as e:
                logging.error(f"Warning: Failed to regenerate analysis after upload: {e}")

        return jsonify({
            "success": True,
            "message": f"Successfully uploaded: {', '.join(uploaded_files)}"
        })
        
    except Exception as e:
        logging.error(f"Error in upload endpoint: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Upload failed: {str(e)}"
        }), 500

def main():
    # Check environment variables
    if not check_env_vars():
        print("Environment check failed. Exiting...")
        return
    
    # Start the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == "__main__":
    main()