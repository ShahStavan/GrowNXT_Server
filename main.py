from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from config import OUTPUT_FOLDER, LOG_DIR, GEMINI_API_KEY
from logger_config import setup_logger
from stock_analyzer import StockAnalyzer
import os
import json

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

OUTPUT_FOLDER.mkdir(exist_ok=True, parents=True)
LOG_DIR.mkdir(exist_ok=True, parents=True)

logger = setup_logger('stock_analyzer', LOG_DIR)

def get_financial_data(stock_folder):
    financial_files = {
        'balance_sheet': 'balance_sheet.json',
        'cash_flow': 'cash_flow.json',
        'profit_loss': 'profit_loss.json',
        'quarterly_results': 'quarterly_results.json',
        'ratios': 'ratios.json',
        'shareholding': 'shareholding.json'
    }
    
    data = {}
    for key, filename in financial_files.items():
        file_path = stock_folder / filename
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                data[key] = json.load(f)
    return data

@app.route('/api/analyze-stock', methods=['POST'])
def analyze_stock():
    """Endpoint to analyze a stock's latest quarterly report"""
    try:
        data = request.get_json()
        stock_name = data.get('stockName')
        stock_folder = OUTPUT_FOLDER / stock_name.lower()
        analysis_path = stock_folder / 'stock_data.json'
        
        if not stock_name:
            logger.error("Stock name not provided")
            return jsonify({'error': 'Stock name is required'}), 400

        logger.info(f"Starting analysis for stock: {stock_name}")
        
        # Get analysis data
        if not analysis_path.exists():
            analyzer = StockAnalyzer(
                stock_name=stock_name,
                output_dir=OUTPUT_FOLDER,
                api_key=GEMINI_API_KEY,
                logger=logger
            )
            analysis_data = analyzer.run_analysis()
        else:
            with open(analysis_path, 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)
        
        # Get financial data
        financial_data = get_financial_data(stock_folder)
        
        # Merge financial data into analysis data instead of nesting
        analysis_data['financials'] = financial_data
            
        response = jsonify(analysis_data)  # Send analysis_data directly
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        return response
        
    except Exception as e:
        logger.error(f"Error getting stock analysis: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)