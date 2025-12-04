from pathlib import Path
from api.search import StockSearch
import os
from dotenv import load_dotenv
from core.config import REQUIRED_ENV_VARS, DATA_DIR
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from services.analysis_service import generate_financial_analysis, generate_dcf_analysis
import requests

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

def check_env():
    """Check required env vars"""
    load_dotenv()
    missing = [v for v in REQUIRED_ENV_VARS if not os.getenv(v)]
    if missing:
        print(f"Missing env vars: {', '.join(missing)}")
        return False
    return True

@app.route('/api/search', methods=['GET'])
def search_stocks():
    q = request.args.get('q', '')
    if not q or len(q) < 3:
        return jsonify([])
    
    DATA_DIR.mkdir(exist_ok=True)
    searcher = StockSearch(DATA_DIR)
    return jsonify(searcher.instant_search(q))

@app.route('/api/stock/save', methods=['POST'])
def save_stock():
    data = request.json
    if not data:
        return jsonify({"success": False, "error": "No data"}), 400
    
    ticker = data.get('ticker') or data.get('symbol')
    if not ticker:
        return jsonify({"success": False, "error": "No ticker"}), 400
    
    DATA_DIR.mkdir(exist_ok=True)
    searcher = StockSearch(DATA_DIR)
    
    if not searcher.save_stock_data(data):
        return jsonify({"success": False, "error": "Save failed"}), 500
    
    return jsonify({"success": True, "message": "Saved"})

@app.route('/api/stocks/<symbol>', methods=['GET'])
def get_stock_data(symbol):
    try:
        folder = DATA_DIR / symbol.lower()
        sdata = folder / 'sData.json'
        
        if not sdata.exists():
            return jsonify({"success": False, "error": f"{symbol} not found"}), 404
        
        with open(sdata, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Load additional files
        files = {
            'balance_sheet': 'balancesheet.json',
            'balance_growth': 'balGrowth.json',
            'quarterly': 'quarterly.json',
            'annual': 'annual.json'
        }
        
        for key, fname in files.items():
            path = folder / fname
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    data[key] = json.load(f)
        
        return jsonify(data)
        
    except Exception as e:
        print(f"Get data failed: {e}")
        return jsonify({"success": False, "error": "Fetch failed"}), 500

@app.route('/api/stocks/<symbol>/analysis', methods=['GET'])
def get_stock_analysis(symbol):
    try:
        folder = DATA_DIR / symbol.lower()
        if not folder.exists():
            return jsonify({"success": False, "error": f"{symbol} not found"}), 404
        
        report = folder / 'report.md'
        mapping = Path(__file__).parent / 'mapping.json'
        
        # Use existing or generate new
        if report.exists():
            with open(report, 'r', encoding='utf-8') as f:
                txt = f.read()
        else:
            txt = generate_financial_analysis(folder, mapping)
        
        # Clean markdown blocks
        if txt.strip().startswith("```markdown"):
            txt = txt.split("```markdown", 1)[1].rsplit("```", 1)[0]
        
        return jsonify({"success": True, "analysis": txt.strip()})
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        return jsonify({"success": False, "error": "Analysis failed"}), 500

@app.route('/api/stocks/<symbol>/dcf', methods=['GET'])
def get_stock_dcf_analysis(symbol):
    try:
        folder = DATA_DIR / symbol.lower()
        if not folder.exists():
            return jsonify({"success": False, "error": f"{symbol} not found"}), 404
        
        report = folder / 'dcf_report.md'
        
        # Use existing or generate new
        if report.exists():
            with open(report, 'r', encoding='utf-8') as f:
                txt = f.read()
        else:
            txt = generate_dcf_analysis(folder)
        
        # Clean markdown blocks
        if txt.strip().startswith("```markdown"):
            txt = txt.split("```markdown", 1)[1].rsplit("```", 1)[0]
        
        return jsonify({"success": True, "analysis": txt.strip()})
        
    except Exception as e:
        print(f"DCF failed: {e}")
        return jsonify({"success": False, "error": "DCF failed"}), 500

@app.route('/api/stocks/<symbol>/upload', methods=['POST'])
def upload_stock_files(symbol):
    try:
        folder = DATA_DIR / symbol.lower()
        folder.mkdir(exist_ok=True)
        mapping = Path(__file__).parent / 'mapping.json'
        files = []

        # Download from URL
        if 'annual_url' in request.form:
            url = request.form['annual_url'].strip()
            if url:
                try:
                    headers = BSE_HEADERS if 'bseindia.com' in url else None
                    res = requests.get(url, stream=True, timeout=30, headers=headers)
                    
                    if res.ok:
                        path = folder / 'annual_report.pdf'
                        size = 0
                        with open(path, 'wb') as f:
                            for chunk in res.iter_content(8192):
                                if chunk:
                                    f.write(chunk)
                                    size += len(chunk)
                        
                        if size > 0:
                            files.append('annual_report.pdf')
                        else:
                            return jsonify({"success": False, "error": "Empty file"}), 400
                    else:
                        err = f"Download failed: {res.status_code}"
                        return jsonify({"success": False, "error": err}), 400
                except Exception as e:
                    return jsonify({"success": False, "error": f"Download error: {e}"}), 500

        # Upload files
        if 'annual' in request.files:
            request.files['annual'].save(str(folder / 'annual_report.pdf'))
            files.append('annual_report.pdf')
            
        if 'presentation' in request.files:
            request.files['presentation'].save(str(folder / 'presentation.pdf'))
            files.append('presentation.pdf')
        
        if 'presentation_url' in request.form and request.form['presentation_url'].strip():
            url = request.form['presentation_url'].strip()
            try:
                headers = BSE_HEADERS if 'bseindia.com' in url else None
                res = requests.get(url, stream=True, timeout=30, headers=headers)
                if res.ok:
                    with open(folder / 'presentation.pdf', 'wb') as f:
                        for chunk in res.iter_content(8192):
                            if chunk:
                                f.write(chunk)
                    files.append('presentation.pdf')
            except Exception as e:
                print(f"Presentation download failed: {e}")

        # Regenerate analysis
        if files:
            try:
                generate_financial_analysis(folder, mapping)
            except Exception as e:
                print(f"Analysis regen failed: {e}")

        return jsonify({"success": True, "message": f"Uploaded: {', '.join(files)}"})
        
    except Exception as e:
        print(f"Upload failed: {e}")
        return jsonify({"success": False, "error": "Upload failed"}), 500

def main():
    if not check_env():
        print("Env check failed")
        return
    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == "__main__":
    main()