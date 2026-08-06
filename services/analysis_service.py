"""Financial Analysis and DCF Report Generation Service Adapter.

Orchestrates fundamental financial report generation via SelfRAGReportGraph
and DCF valuation routines using live Vercel REST tools and targeted RAG vector chunks.

Google Python Style Guide Compliant.
"""

import logging
from pathlib import Path
from typing import Optional, Union

from core.config import MAPPING_FILE_PATH
from core.llm_config import DEFAULT_MODEL, create_client, create_content_part, read_file_content
from core.prompts import DCF_PROMPT
from services.financial_tools import FinancialDataAgent

logger = logging.getLogger(__name__)


def generate_financial_analysis(folder: Union[str, Path], mapping: Optional[Path] = None) -> str:
    """Generates financial report using Advanced Self-RAG Graph Pipeline with live REST tools.

    Args:
        folder (Union[str, Path]): Stock directory path.
        mapping (Optional[Path]): Metric mapping dictionary file path.

    Returns:
        str: Generated Markdown analysis report text string.
    """
    folder_path = Path(folder)
    symbol = folder_path.name.upper()

    try:
        from services.graph_pipeline import SelfRAGReportGraph
        logger.info("Executing Advanced Self-RAG Report Pipeline for symbol '%s' at folder: %s", symbol, folder_path)
        graph_pipeline = SelfRAGReportGraph(symbol=symbol, folder_path=folder_path)
        return graph_pipeline.execute_pipeline()

    except Exception as exc:
        logger.error("Self-RAG Pipeline execution failed for symbol %s: %s", symbol, exc, exc_info=True)
        raise


def generate_dcf_analysis(folder: Union[str, Path]) -> str:
    """Generates Discounted Cash Flow (DCF) valuation report fetching live data from Vercel API.

    Args:
        folder (Union[str, Path]): Stock data directory path.

    Returns:
        str: Generated DCF valuation report text in Markdown format.
    """
    folder_path = Path(folder)
    symbol = folder_path.name.upper()
    agent = FinancialDataAgent()

    try:
        logger.info("Generating DCF Valuation Report live for symbol '%s'...", symbol)
        client = create_client()
        model = client.GenerativeModel(DEFAULT_MODEL)
        parts = []

        # 1. Add ground-truth metric dictionary mapping if present
        mapping_file = MAPPING_FILE_PATH
        if mapping_file.exists():
            mapping_content = read_file_content(mapping_file)
            if mapping_content:
                parts.append(create_content_part(mapping_content))

        # 2. Fetch ground-truth statement data live from Vercel REST endpoints
        statement_bundle = agent.get_context_for_section("financial_performance", symbol)
        solvency_bundle = agent.get_context_for_section("solvency_analysis", symbol)

        if statement_bundle:
            parts.append(create_content_part(statement_bundle))
        if solvency_bundle:
            parts.append(create_content_part(solvency_bundle))

        parts.append(create_content_part(DCF_PROMPT))

        prompt_text = "\n\n".join([p["text"] for p in parts if "text" in p])
        response = model.generate_content(prompt_text)

        report_text = response.text if response and response.text else "DCF valuation calculation failed."
        dcf_report_file = folder_path / "dcf_report.md"

        folder_path.mkdir(parents=True, exist_ok=True)
        with open(dcf_report_file, "w", encoding="utf-8") as f_out:
            f_out.write(report_text)

        logger.info("Successfully generated and saved DCF valuation report for %s to %s", symbol, dcf_report_file)
        return report_text

    except Exception as exc:
        logger.error("DCF analysis generation failed for symbol %s: %s", symbol, exc, exc_info=True)
        raise
