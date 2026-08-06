from pathlib import Path
import time
from core.llm_config import create_client, create_content_part, read_file_content, DEFAULT_MODEL
from core.prompts import ANALYSIS_PROMPT, DCF_PROMPT


def _read_json_files(folder: Path) -> list:
    """Read all JSON files"""
    files = [
        'qtGrowth.json', 'balGrowth.json', 'balancesheet.json',
        'annual.json', 'anGrowth.json', 'quarterly.json',
        'sData.json', 'cashflow.json', 'summary.json'
    ]
    
    parts = []
    for f in files:
        path = folder / f
        if path.exists():
            content = read_file_content(str(path))
            parts.append(create_content_part(content))
    return parts


def _read_pdf_files(folder: Path) -> list:
    """Read PDF files with retry"""
    files = ['annual_report.pdf', 'presentation.pdf']
    parts = []
    
    for f in files:
        path = folder / f
        if not path.exists():
            continue
        
        # Retry logic
        for i in range(3):
            try:
                content = read_file_content(str(path))
                if content and content.strip():
                    parts.append(create_content_part(f"Analysis from {f}:\n{content}"))
                    break
                time.sleep(1)
            except Exception as e:
                if i == 2:
                    print(f"Failed {f}: {e}")
    
    return parts


def generate_financial_analysis(folder: Path, mapping: Path = None) -> str:
    """Generate financial analysis report using Advanced Self-RAG Graph Pipeline."""
    try:
        from services.graph_pipeline import SelfRAGReportGraph
        print(f"Executing Advanced Self-RAG Report Pipeline for folder: {folder}")
        graph_pipeline = SelfRAGReportGraph(folder)
        return graph_pipeline.execute_pipeline()
    except Exception as e:
        print(f"Self-RAG Pipeline failed, falling back to standard generator: {e}")
        client = create_client()
        model = client.GenerativeModel(DEFAULT_MODEL)
        
        # Collect content fallback
        parts = []
        parts.extend(_read_json_files(folder))
        parts.extend(_read_pdf_files(folder))
        
        if mapping and mapping.exists():
            parts.append(create_content_part(read_file_content(str(mapping))))
        
        parts.append(ANALYSIS_PROMPT)
        res = model.generate_content("\n".join(parts))
        
        report = folder / 'report.md'
        with open(report, 'w', encoding='utf-8', errors='ignore') as f:
            f.write(res.text)
        
        return res.text

def generate_dcf_analysis(folder: Path) -> str:
    """Generate DCF analysis report"""
    try:
        client = create_client()
        model = client.GenerativeModel(DEFAULT_MODEL)
        
        required = ['annual.json', 'sData.json', 'cashflow.json', 'balancesheet.json']
        parts = []
        missing = []
        
        # Add mapping
        mapping = Path(__file__).parent / 'mapping.json'
        if mapping.exists():
            parts.append(create_content_part(read_file_content(str(mapping))))
        
        # Add required files
        for f in required:
            path = folder / f
            if path.exists():
                parts.append(create_content_part(read_file_content(str(path))))
            else:
                missing.append(f)
        
        if missing:
            raise FileNotFoundError(f"Missing: {', '.join(missing)}")
        
        # Add prompt and generate
        parts.append(DCF_PROMPT)
        res = model.generate_content("\n".join(parts))
        
        # Save
        report = folder / 'dcf_report.md'
        with open(report, 'w', encoding='utf-8') as f:
            f.write(res.text)
        
        return res.text
        
    except Exception as e:
        print(f"DCF failed: {e}")
        raise
