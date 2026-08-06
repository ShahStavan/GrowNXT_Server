"""Flask Web Application API Server for GrowNXT Fundamental Financial Platform.

Provides RESTful endpoints for stock discovery, live statement data retrieval,
Self-RAG financial report generation, DCF valuation, and document filing uploads.

Google Python Style Guide Compliant.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Tuple, Union
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
import requests

from api.search import StockSearch
from core.config import DATA_DIR, MAPPING_FILE_PATH, REQUIRED_ENV_VARS
from services.analysis_service import generate_dcf_analysis, generate_financial_analysis
from services.financial_tools import fetch_full_financial_bundle_tool, fetch_stock_summary_tool

logger = logging.getLogger(__name__)

# Standard Headers for BSE/NSE Document Downloader
BSE_HEADERS: Dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
    "Referer": "https://www.bseindia.com/",
    "Upgrade-Insecure-Requests": "1",
}


def check_env() -> bool:
    """Verifies that all required environment variables are present before starting.

    Returns:
        bool: True if environment validation passes, False otherwise.
    """
    load_dotenv()
    missing_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
    if missing_vars:
        logger.error("Environment verification failed. Missing required variables: %s", ", ".join(missing_vars))
        return False
    logger.info("Environment variable check passed successfully.")
    return True


def create_app() -> Flask:
    """Application factory for constructing and initializing the Flask application instance.

    Returns:
        Flask: Fully configured Flask application instance.
    """
    app = Flask(__name__)

    # Centralized CORS setup for API routes
    CORS(
        app,
        resources={
            r"/api/*": {
                "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
                "methods": ["GET", "POST", "OPTIONS"],
                "allow_headers": ["Content-Type", "Authorization"],
            }
        },
    )

    # -------------------------------------------------------------------------
    # Route Handlers
    # -------------------------------------------------------------------------

    @app.route("/api/search", methods=["GET"])
    def search_stocks() -> Tuple[Any, int]:
        """Searches for stock tickers given a query parameter 'q'."""
        query_param = request.args.get("q", "").strip()
        if not query_param or len(query_param) < 3:
            return jsonify([]), 200

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        searcher = StockSearch(DATA_DIR)
        results = searcher.instant_search(query_param)
        return jsonify(results), 200

    @app.route("/api/stock/save", methods=["POST"])
    def save_stock() -> Tuple[Any, int]:
        """Saves stock metadata and verifies ticker accessibility."""
        payload = request.get_json(silent=True)
        if not payload:
            return jsonify({"success": False, "error": "Request body must contain valid JSON data"}), 400

        ticker = payload.get("ticker") or payload.get("symbol")
        if not ticker:
            return jsonify({"success": False, "error": "Ticker symbol is required"}), 400

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        searcher = StockSearch(DATA_DIR)

        if not searcher.save_stock_data(payload):
            return jsonify({"success": False, "error": f"Failed to save financial data for {ticker}"}), 500

        return jsonify({"success": True, "message": f"Successfully registered stock ticker {ticker}"}), 200

    @app.route("/api/stocks/<symbol>", methods=["GET"])
    def get_stock_data(symbol: str) -> Tuple[Any, int]:
        """Retrieves aggregated JSON financial statements live from Vercel REST service."""
        try:
            clean_symbol = symbol.strip().upper()
            logger.info("Fetching live financial statement bundle for symbol '%s' via Vercel REST API...", clean_symbol)

            bundle_json_str = fetch_full_financial_bundle_tool.invoke({"symbol": clean_symbol})
            summary_json_str = fetch_stock_summary_tool.invoke({"symbol": clean_symbol})

            if "unavailable for stock" in bundle_json_str and "unavailable for stock" in summary_json_str:
                return jsonify({"success": False, "error": f"Stock data for '{symbol}' not found on Vercel service"}), 404

            combined_data = {}

            # Parse summary profile if available
            try:
                summary_obj = json.loads(summary_json_str)
                if isinstance(summary_obj, dict):
                    combined_data["info"] = summary_obj
            except Exception:
                pass

            # Parse financial statement bundle
            try:
                bundle_obj = json.loads(bundle_json_str)
                if isinstance(bundle_obj, dict):
                    combined_data.update(bundle_obj)
            except Exception:
                pass

            combined_data["ticker"] = clean_symbol
            return jsonify(combined_data), 200

        except Exception as exc:
            logger.error("Failed retrieving live stock data for symbol %s: %s", symbol, exc, exc_info=True)
            return jsonify({"success": False, "error": "Failed to fetch stock financial data"}), 500

    @app.route("/api/stocks/<symbol>/analysis", methods=["GET"])
    def get_stock_analysis(symbol: str) -> Tuple[Any, int]:
        """Generates or retrieves Self-RAG financial analysis report for a stock."""
        try:
            folder = DATA_DIR / symbol.strip().lower()
            folder.mkdir(parents=True, exist_ok=True)
            report_file = folder / "report.md"

            if report_file.exists():
                with open(report_file, "r", encoding="utf-8") as f_handle:
                    report_text = f_handle.read()
            else:
                report_text = generate_financial_analysis(folder, MAPPING_FILE_PATH)

            # Clean markdown code block wraps if present
            cleaned_text = report_text.strip()
            if cleaned_text.startswith("```markdown"):
                cleaned_text = cleaned_text.split("```markdown", 1)[1].rsplit("```", 1)[0]
            elif cleaned_text.startswith("```"):
                cleaned_text = cleaned_text.split("```", 1)[1].rsplit("```", 1)[0]

            return jsonify({"success": True, "analysis": cleaned_text.strip()}), 200

        except Exception as exc:
            logger.error("Analysis generation failed for symbol %s: %s", symbol, exc, exc_info=True)
            return jsonify({"success": False, "error": "Financial analysis generation failed"}), 500

    @app.route("/api/stocks/<symbol>/dcf", methods=["GET"])
    def get_stock_dcf_analysis(symbol: str) -> Tuple[Any, int]:
        """Generates or retrieves Discounted Cash Flow (DCF) valuation report for a stock."""
        try:
            folder = DATA_DIR / symbol.strip().lower()
            folder.mkdir(parents=True, exist_ok=True)

            dcf_report_file = folder / "dcf_report.md"

            if dcf_report_file.exists():
                with open(dcf_report_file, "r", encoding="utf-8") as f_handle:
                    report_text = f_handle.read()
            else:
                report_text = generate_dcf_analysis(folder)

            # Clean markdown code blocks
            cleaned_text = report_text.strip()
            if cleaned_text.startswith("```markdown"):
                cleaned_text = cleaned_text.split("```markdown", 1)[1].rsplit("```", 1)[0]
            elif cleaned_text.startswith("```"):
                cleaned_text = cleaned_text.split("```", 1)[1].rsplit("```", 1)[0]

            return jsonify({"success": True, "analysis": cleaned_text.strip()}), 200

        except Exception as exc:
            logger.error("DCF analysis failed for symbol %s: %s", symbol, exc, exc_info=True)
            return jsonify({"success": False, "error": "DCF analysis calculation failed"}), 500

    @app.route("/api/stocks/<symbol>/upload", methods=["POST"])
    def upload_stock_files(symbol: str) -> Tuple[Any, int]:
        """Handles PDF annual report downloads and manual file uploads for RAG indexing."""
        try:
            folder = DATA_DIR / symbol.strip().lower()
            folder.mkdir(parents=True, exist_ok=True)
            saved_files = []

            # 1. Process Annual Report URL Download
            annual_url = request.form.get("annual_url", "").strip()
            if annual_url:
                try:
                    headers = BSE_HEADERS if "bseindia.com" in annual_url else None
                    response = requests.get(annual_url, stream=True, timeout=30, headers=headers)
                    if response.ok:
                        pdf_path = folder / "annual_report.pdf"
                        downloaded_bytes = 0
                        with open(pdf_path, "wb") as f_out:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f_out.write(chunk)
                                    downloaded_bytes += len(chunk)

                        if downloaded_bytes > 0:
                            saved_files.append("annual_report.pdf")
                        else:
                            return jsonify({"success": False, "error": "Downloaded annual report file was empty"}), 400
                    else:
                        return jsonify({"success": False, "error": f"Download failed with status {response.status_code}"}), 400
                except Exception as exc:
                    logger.error("Failed downloading annual report from URL %s: %s", annual_url, exc)
                    return jsonify({"success": False, "error": f"Download exception: {str(exc)}"}), 500

            # 2. Process Direct File Uploads
            if "annual" in request.files:
                request.files["annual"].save(str(folder / "annual_report.pdf"))
                saved_files.append("annual_report.pdf")

            if "presentation" in request.files:
                request.files["presentation"].save(str(folder / "presentation.pdf"))
                saved_files.append("presentation.pdf")

            # 3. Process Presentation URL Download
            presentation_url = request.form.get("presentation_url", "").strip()
            if presentation_url:
                try:
                    headers = BSE_HEADERS if "bseindia.com" in presentation_url else None
                    response = requests.get(presentation_url, stream=True, timeout=30, headers=headers)
                    if response.ok:
                        with open(folder / "presentation.pdf", "wb") as f_out:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f_out.write(chunk)
                        saved_files.append("presentation.pdf")
                except Exception as exc:
                    logger.warning("Presentation download exception: %s", exc)

            # Regenerate analysis report if files were updated
            if saved_files:
                try:
                    generate_financial_analysis(folder, MAPPING_FILE_PATH)
                except Exception as exc:
                    logger.warning("Failed auto-regenerating analysis after file upload: %s", exc)

            return jsonify({
                "success": True,
                "message": f"Successfully processed files: {', '.join(saved_files)}" if saved_files else "No files processed"
            }), 200

        except Exception as exc:
            logger.error("File upload route failed for symbol %s: %s", symbol, exc, exc_info=True)
            return jsonify({"success": False, "error": "File upload processing failed"}), 500

    # -------------------------------------------------------------------------
    # Error Handlers
    # -------------------------------------------------------------------------

    @app.errorhandler(404)
    def handle_not_found(error: Any) -> Tuple[Any, int]:
        return jsonify({"success": False, "error": "Requested API route not found"}), 404

    @app.errorhandler(500)
    def handle_server_error(error: Any) -> Tuple[Any, int]:
        return jsonify({"success": False, "error": "Internal server error occurred"}), 500

    return app


app: Flask = create_app()


def main() -> None:
    """Server entry point execution handler."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    if not check_env():
        logger.error("Stopping application startup due to missing environment variables.")
        return

    logger.info("Starting GrowNXT Flask API Server on port 5000...")
    app.run(host="0.0.0.0", port=5000, debug=True)


if __name__ == "__main__":
    main()