"""
Dynamic Prompt Registry for Advanced Financial Analyst Report Generator.

Defines modular prompts in plain English for:
- Company Overview & Profile
- Core Business Operations & Verticals
- Strategic Expansion Plans & Capex Projects
- Key Clients, Suppliers & Market Footprint
- Financial Results & Growth Metrics
- DuPont Analysis & Return Ratios (ROE / ROCE)
- Balance Sheet & Solvency Analysis
- Financial Strengths & Weaknesses
- Self-RAG Evaluation & Critiques
"""

from typing import Dict, Any, Optional

COMPANY_OVERVIEW_PROMPT = """
Target: Comprehensive Company Profile & Background Analysis

Task:
Analyze the provided financial context and annual filings to construct a clear, easy-to-understand corporate profile.

Requirements:
1. Corporate Summary: Name, Ticker, Sector, Industry, Market Capitalization.
2. Market Standing: Industry leadership, key brands, and corporate structure.
3. Summary Table:
   | Profile Field | Details |
   | :--- | :--- |
   | Company Name | [Full Name] |
   | Symbol | [Ticker] |
   | Sector | [Sector Name] |
   | Industry | [Industry Name] |
   | Market Cap | [Market Cap in Cr] |

Format under an `### Executive & Company Overview` header.
"""

COMPANY_OPERATIONS_PROMPT = """
Target: Core Business Operations & Primary Verticals

Task:
Detail the company's business model and operating divisions in simple, plain English terms that any everyday investor can understand.

Requirements:
1. Revenue Engine: Explain in simple terms how the company makes money across its divisions.
2. Operating Verticals: List primary business segments (e.g., Airports, Energy, Mining, Data Centers).
3. Investor Takeaway: Provide a bulleted "💡 Simple Summary for Investors" box explaining the business model.

Format under an `### Core Business Operations & Operating Verticals` header.
"""

EXPANSION_PLANS_PROMPT = """
Target: Strategic Expansion Plans & Ongoing Capex Projects

Task:
Examine ongoing expansion plans, strategic projects, and capital deployment in simple terms.

Requirements:
1. Ongoing Projects: List new factories, airports, or facilities being built.
2. Investor Takeaway: Provide a "💡 Simple Summary for Investors" explaining what investors should watch for as these new projects open.

Format under an `### Strategic Expansion Plans & Capex Pipeline` header.
"""

CLIENTS_MARKET_FOOTPRINT_PROMPT = """
Target: Key Clients, Customer Segments & Market Footprint

Task:
Explain the company's client base, government contracts, and competitive advantages in simple terms.

Requirements:
1. Concessions & Contracts: Highlight long-term government contracts and major enterprise partners.
2. Investor Takeaway: Provide a "💡 Simple Summary for Investors" explaining the company's competitive advantage (economic moat).

Format under an `### Key Clients, Concessions & Market Footprint` header.
"""

FINANCIAL_RESULTS_PROMPT = """
Target: Financial Results & Growth Metrics Analysis

Task:
Examine the quarterly and annual financial tables (latest data first) and provide a plain-English explanation of performance.

Requirements:
1. Quarterly Results Table (in ₹ Cr): Display latest 4 quarters with Revenue, Operating Profit, Net Profit, and Sales Trend.
2. Annual Results Table (in ₹ Cr): Display latest 4 fiscal years with Revenue, Operating Profit, Net Profit, and YoY Growth.
3. Simple Investor Insights:
   - **Sales & Revenue**: Plain-English explanation of sales trend.
   - **Operating Profit**: Plain-English explanation of operating efficiency.
   - **Net Profit**: Plain-English explanation of take-home earnings.

Format under an `### Financial Results & Growth Performance` header.
"""

DUPONT_ANALYSIS_PROMPT = """
Target: DuPont Analysis & Return Ratios (ROE / ROCE Decomposition)

Task:
Examine the ground-truth balance sheet and profit loss data to calculate DuPont ROE components and ROCE with strict mathematical precision.

Requirements:
1. DuPont ROE Decomposition Formula Table:
   $$\\text{{ROE}} = \\text{{Net Profit Margin}} \\times \\text{{Asset Turnover}} \\times \\text{{Financial Leverage}}$$
   | DuPont Driver | Calculation Formula | Value (%) / Ratio | Analyst Interpretation |
   | :--- | :--- | :--- | :--- |
   | Net Profit Margin | PAT / Revenue | [Val%] | Profit generated per rupee of sales |
   | Asset Turnover | Revenue / Total Assets | [Valx] | Revenue generated per rupee of assets |
   | Financial Leverage | Total Assets / Equity | [Valx] | Equity multiplier from debt capital |
   | **Return on Equity (ROE)** | **PAT / Equity** | **[Val%]** | **Overall return earned on shareholder equity** |

2. Return on Capital Employed (ROCE) Summary Table:
   | Metric | Calculation Formula | Value (%) | Interpretation |
   | :--- | :--- | :--- | :--- |
   | **ROCE** | **EBIT / (Equity + Debt)** | **[Val%]** | **Return generated across all long-term capital** |

3. Simple Investor Takeaway:
   Provide a "💡 Simple Summary for Investors" box explaining whether profits are coming from high profit margins, fast product sales, or borrowed money (leverage).

Format under a `### DuPont Analysis & Return Ratios (ROE / ROCE)` header.
"""

BALANCE_SHEET_PROMPT = """
Target: Balance Sheet Data & Solvency Analysis

Task:
Analyze balance sheet health, debt levels, equity, and cash reserves in simple investor terms.

Requirements:
1. Capital Structure Table: Display Net Worth, Total Debt, Bank Cash, and Debt-to-Equity ratio.
2. Simple Investor Insights:
   - **Debt Level**: Plain-English explanation of borrowing and debt safety.
   - **Cash Buffer**: Plain-English explanation of cash reserves and loan repayment ability.

Format under an `### Balance Sheet & Solvency Analysis` header.
"""

STRENGTHS_WEAKNESSES_PROMPT = """
Target: Financial Performance Strengths & Risk Factors

Task:
Summarize the key Bull case strengths and Bear case vulnerabilities in clear, simple bullet points.

Requirements:
1. Bull Case Strengths 📈: 2 key strengths explained simply.
2. Bear Case Vulnerabilities 📉: 2 key risks explained simply.

Format under an `### Financial Strengths & Risk Factors` header.
"""

SELF_RAG_CRITIQUE_PROMPT = """
Target: Self-RAG Factuality & Quality Verification

Task:
Review the generated section against ground truth context.

Draft Section To Review:
{draft_section}

Ground Truth Context Data:
{ground_truth_context}

Respond ONLY in valid JSON format:
{{
    "passed": true,
    "score": 1.0,
    "feedback": "Section factual verification passed",
    "refined_content": ""
}}
"""


class DynamicPromptRegistry:
    """Registry to select and format dynamic prompts for report generation steps."""

    @staticmethod
    def get_prompt(prompt_type: str, **kwargs) -> str:
        registry = {
            "company_overview": COMPANY_OVERVIEW_PROMPT,
            "company_operations": COMPANY_OPERATIONS_PROMPT,
            "expansion_plans": EXPANSION_PLANS_PROMPT,
            "clients_market": CLIENTS_MARKET_FOOTPRINT_PROMPT,
            "financial_results": FINANCIAL_RESULTS_PROMPT,
            "dupont_analysis": DUPONT_ANALYSIS_PROMPT,
            "balance_sheet": BALANCE_SHEET_PROMPT,
            "strengths_weaknesses": STRENGTHS_WEAKNESSES_PROMPT,
            "self_rag_critique": SELF_RAG_CRITIQUE_PROMPT,
        }
        
        prompt_template = registry.get(prompt_type)
        if not prompt_template:
            raise ValueError(f"Unknown prompt_type: '{prompt_type}'. Valid options: {list(registry.keys())}")
        
        if kwargs:
            return prompt_template.format(**kwargs)
        return prompt_template
