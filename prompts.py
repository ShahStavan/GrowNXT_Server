FINANCIAL_ANALYSIS_GUIDELINES = """
FINANCIAL ANALYSIS GUIDELINES [For Each Section]:

BALANCE SHEET ANALYSIS:
Strengths to look for:
- Strong asset base growth (>15% YoY)
- Low debt-to-equity ratio (<2)
- Healthy current ratio (>1.5)
- Strong cash reserves (>10% of assets)
- Quality of assets (high fixed asset turnover)
- Working capital efficiency
- Asset monetization potential
- Strong capital structure

Weaknesses to watch:
- High leverage ratios
- Poor working capital management
- Declining asset quality
- High goodwill/intangibles
- Contingent liabilities
- Low liquidity ratios
- High debt servicing costs
- Asset-liability mismatches

PROFIT & LOSS ANALYSIS:
Strengths indicators:
- Consistent revenue growth (>15% CAGR)
- Expanding margins
- Cost optimization success
- High operating leverage benefits
- Diversified revenue streams
- Strong order book/backlog
- Pricing power
- Operating efficiency improvements

Weaknesses red flags:
- Revenue concentration risk
- Margin pressure points
- High fixed cost structure
- Forex vulnerabilities
- Raw material price sensitivity
- Customer concentration
- Seasonal dependencies
- High operating costs

QUARTERLY PERFORMANCE:
Strengths to highlight:
- Sequential growth momentum
- Margin expansion trends
- Market share gains
- New customer additions
- Product mix improvements
- Geographic expansion success
- Order book growth
- Working capital optimization

Weaknesses to note:
- Revenue volatility
- Margin compression
- Market share losses
- Customer churn
- Product obsolescence risk
- Geographic concentration
- Order book decline
- Working capital stress

RATIO ANALYSIS:
Positive indicators:
- Industry-leading ROE/ROCE
- Strong interest coverage
- Healthy asset turnover
- Efficient working capital cycle
- Low cash conversion cycle
- Strong free cash flow yield
- Quality of earnings
- Capital allocation efficiency

Negative indicators:
- Below-industry returns
- Poor capital efficiency
- Extended working capital cycle
- Weak cash flow metrics
- Poor dividend payout
- High capital intensity
- Low return on capital
- Poor cash conversion

SHAREHOLDING ANALYSIS:
Strengths to consider:
- High promoter holding
- Increasing institutional interest
- Strong institutional backing
- Low pledge percentage
- Regular promoter buying
- Quality institutional investors
- Long-term investor base
- Strategic investor presence

Concerns to highlight:
- Declining promoter stake
- High pledge percentage
- Institutional selling
- High retail concentration
- Ownership concentration
- Poor free float
- Frequent block deals
- Quality of investors
"""

COMPANY_OVERVIEW_PROMPT = """Analyze this investor presentation from a professional financial analyst's perspective. 
Consider all aspects that would be important for investment decisions.

1. COMPANY OVERVIEW [Extract a concise, clear description]:
   - Core business activities and main revenue streams
   - Key products/services offered
   - Geographic presence and market coverage
   - Target customer segments
   - Industry position and market share

Return Company overview in bullet points and append those points in list format. 
Note:
No introductory sentences needed.
Keep the response concise and to the point.
Keep the response professional and investment-focused.

Return the clean json which returns the list of points
Don't explain anything just return the given json format response
Format in this manner:

shortDescription: [
  point 1,
  point 2,
  so on
]
"""

COMPANY_OVERVIEW_PROMPT_HINDI = """तुम एक भारतीय वित्तीय विश्लेषक हो और तुम्हें इस कंपनी का मूल्यांकन हिंदी में करना है।
तुम्हारा विश्लेषण हिंदी भाषी निवेशकों के लिए है।

सटीक विश्लेषण करें:
1. मुख्य व्यवसाय:
   - कंपनी का मुख्य कार्य क्या है?
   - राजस्व के प्रमुख स्रोत कौन से हैं?
   - कौन से उत्पाद या सेवाएं प्रदान करते हैं?

2. बाजार उपस्थिति:
   - किन क्षेत्रों में काम करते हैं?
   - विस्तार की योजनाएं क्या हैं?
   - प्रमुख ग्राहक कौन हैं?

3. व्यावसायिक स्थिति:
   - बाजार में कंपनी की स्थिति
   - प्रतिस्पर्धा में कहां खड़े हैं?
   - भविष्य की संभावनाएं क्या हैं?

अनिवार्य दिशा-निर्देश:
1. सभी जानकारी केवल हिंदी में लिखें
2. सरल और स्पष्ट भाषा का प्रयोग करें
3. तकनीकी शब्दों को अंग्रेजी में रख सकते हैं (जैसे EBITDA, ROE आदि)
4. संख्यात्मक डेटा यथावत रखें
5. विश्लेषण व्यावसायिक दृष्टिकोण से होना चाहिए

इस JSON फॉर्मेट में जवाब दें:
{
    "shortDescription": [
        "पहला बिंदु हिंदी में",
        "दूसरा बिंदु हिंदी में",
        "और बिंदु हिंदी में"
    ]
}"""

DETAILED_FINANCIAL_ANALYSIS_PROMPT = """Analyze this investor presentation as a Financial Manager at Vanguard Fund.
Provide detailed analysis for investment decisions using these categories:

1. Business Model & Strategy:
   - Core business model details
   - Revenue drivers and profit margins by segment
   - Strategic priorities and growth initiatives
   - Competitive landscape analysis
   - Pricing strategies impact

2. Financial Performance & Projections:
   - Historical performance trends
   - Key financial ratios analysis
   - Management projections assessment
   - Capital structure analysis
   - Free cash flow implications

3. Operational Capabilities & Risks:
   - Operational efficiency evaluation
   - Key risk identification
   - ESG performance assessment
   - Legal and regulatory considerations

4. Sectorial Analysis:
   - Industry position
   - Market share analysis
   - Growth potential
   - Sector-specific challenges

Note:
No introductory sentences needed.
Keep the response concise and to the point.
Keep the response professional and investment-focused.

Return the clean json which returns the list of points
Don't explain anything just return the given json format response
Format in this manner:

Return analysis in clean JSON format:
{
    "Business Model & Strategy": [
      point 1
      point 2
      so on
    ],
      "Financial Performance & Projections": [
         point 1
         point 2
         so on
      ],
      "Operational Capabilities & Risks": [
         point 1
         point 2
         so on
      ],
      "Sectorial Analysis": [
         point 1
         point 2
         so on
      ]
    
}
Points Guidelines:
Make sure before the point it shouldn't contain { and after completion of point it shouldn't contain } for all section
Make sure point shouldn't begin with point 1: and so on give stratigh to the point 
"""

DETAILED_FINANCIAL_ANALYSIS_PROMPT_HINDI = """आप एक वरिष्ठ वित्तीय विश्लेषक हैं और आपको इस कंपनी का विस्तृत विश्लेषण हिंदी में करना है।
यह विश्लेषण हिंदी भाषी निवेशकों के लिए है।

विश्लेषण के मुख्य बिंदु:

1. व्यवसाय मॉडल और रणनीति:
   - व्यवसाय कैसे काम करता है?
   - आय कैसे कमाते हैं?
   - भविष्य की योजनाएं क्या हैं?
   - प्रतिस्पर्धियों से कैसे अलग हैं?

2. वित्तीय प्रदर्शन:
   - राजस्व और मुनाफे का विश्लेषण
   - मुख्य वित्तीय अनुपात
   - प्रबंधन के अनुमान
   - कंपनी की वित्तीय स्थिति

3. परिचालन क्षमताएं और जोखिम:
   - संचालन में कितने सक्षम हैं?
   - क्या जोखिम हैं?
   - पर्यावरण प्रभाव
   - कानूनी मुद्दे

4. उद्योग विश्लेषण:
   - उद्योग में स्थिति
   - बाजार हिस्सेदारी
   - विकास की संभावनाएं
   - क्षेत्र की चुनौतियां

महत्वपूर्ण निर्देश:
1. विश्लेषण पूरी तरह हिंदी में होना चाहिए
2. हर बिंदु स्पष्ट और विस्तृत होना चाहिए
3. तकनीकी शब्द अंग्रेजी में रख सकते हैं
4. आंकड़े और प्रतिशत यथावत रखें
5. व्यावसायिक दृष्टिकोण बनाए रखें

इस JSON फॉर्मेट में उत्तर दें:
{
    "Business Model & Strategy": [
        "व्यवसाय मॉडल का पहला बिंदु",
        "दूसरा बिंदु"
    ],
    "Financial Performance & Projections": [
        "वित्तीय प्रदर्शन का पहला बिंदु",
        "दूसरा बिंदु"
    ],
    "Operational Capabilities & Risks": [
        "परिचालन का पहला बिंदु",
        "दूसरा बिंदु"
    ],
    "Sectorial Analysis": [
        "क्षेत्रीय विश्लेषण का पहला बिंदु",
        "दूसरा बिंदु"
    ]
}"""

DETAILED_ANALYSIS_PROMPT = """You are a highly sophisticated financial analyst working as a Portfolio Manager for a large, multi-national corporation (MNC) investment firm. Your task is to meticulously analyze the provided documents for detailed financial analysis.

Your analysis should conform to this JSON schema:
{
  "keyFinancialHighlights": {
    "revenueGrowth": number|null|array,
    "EBITDAGrowth": number|null|array,
    "profitAfterTax": number|null,
    "EPS": number|null,
    "netDebtToEquityRatio": number|null|array
  },
  "orderBookInformation": {
    "currentOrderBookValue": number|null|array,
    "orderBookType": string|null,
    "futureOrderPipeline": number|null,
    "orderExecutionTimeline": string|null
  },
  "guidance": {
    "revenueGrowthGuidance": number|null|array,
    "PATMarginGuidance": number|null|array,
    "capitalExpenditureGuidance": number|null
  },
  "businessStrategy": {
    "expansionPlans": array|null,
    "capacityAdditionTarget": number|null,
    "projectTypeFocus": string|null,
    "technologicalAdvancements": string|null,
    "netDebtZeroStatus": boolean|null,
    "sustainabilityCommitment": string|null
  },
  "risksAndChallenges": string|null
}

Return only the JSON object, no additional text."""