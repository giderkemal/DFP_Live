from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any, Optional
import pandas as pd
import os
import json
import asyncio
from datetime import datetime, date
import logging
from pydantic import BaseModel
import anthropic
import xml.etree.ElementTree as ET
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="Field Intelligence Report API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Claude client - will be initialized when needed
claude_client = None

def get_claude_client():
    global claude_client
    if claude_client is None:
        try:
            claude_client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
        except Exception as e:
            logger.error(f"Failed to initialize Claude client: {e}")
            claude_client = None
    return claude_client

# Global data storage
data_cache = {}

class FilterRequest(BaseModel):
    date_range: List[str]
    feedback_class: List[str] = []
    form_type: str = "All"
    region: List[str] = []
    market: List[str] = []
    location: List[str] = []
    tmo: List[str] = []
    brand: List[str] = []
    product_category: List[str] = []
    pmi_product: List[str] = []
    switch_from: List[str] = []
    switch_to: List[str] = []

class ChatMessage(BaseModel):
    content: str
    role: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    system_prompt: str

def load_csv_data():
    """Load the CSV data and cache it"""
    global data_cache
    try:
        # Load the CSV data
        df = pd.read_csv('/app/DFP_DATA.csv')
        
        # Convert date columns
        if 'SUBMISSION_DATETIME' in df.columns:
            df['SUBMISSION_DATETIME'] = pd.to_datetime(df['SUBMISSION_DATETIME'])
        elif 'submission_datetime' in df.columns:
            df['submission_datetime'] = pd.to_datetime(df['submission_datetime'])
            df = df.rename(columns={'submission_datetime': 'SUBMISSION_DATETIME'})
            
        # Standardize column names
        column_mapping = {
            'field_intelligence_translated': 'FIELD_INTELLIGENCE_TRANSLATED',
            'class': 'CLASS',
            'location_name': 'LOCATION_NAME',
            'form_type': 'FORM_TYPE',
            'product_category_name': 'PRODUCT_CATEGORY_NAME',
            'brand_name_from': 'BRAND_NAME_FROM',
            'brand_name_to': 'BRAND_NAME_TO',
            'pmi_product_name': 'PMI_PRODUCT_NAME',
            'tmo_name': 'TMO_NAME',
            'brand_name': 'BRAND_NAME',
            'vp_region_name': 'VP_REGION_NAME',
            'df_market_name': 'DF_MARKET_NAME'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})
        
        data_cache['raw_data'] = df
        logger.info(f"Loaded CSV data with {len(df)} rows and columns: {list(df.columns)}")
        return df
    except Exception as e:
        logger.error(f"Error loading CSV data: {e}")
        return pd.DataFrame()

@app.on_event("startup")
async def startup_event():
    """Initialize data on startup"""
    load_csv_data()

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "data_loaded": len(data_cache.get('raw_data', pd.DataFrame())) > 0}

@app.get("/api/metadata")
async def get_metadata():
    """Get metadata for filters"""
    try:
        df = data_cache.get('raw_data')
        if df is None or df.empty:
            df = load_csv_data()
        
        if df.empty:
            raise HTTPException(status_code=500, detail="No data available")
        
        metadata = {}
        
        # Date range
        if 'SUBMISSION_DATETIME' in df.columns:
            dates = pd.to_datetime(df['SUBMISSION_DATETIME']).dt.date
            metadata['DATE'] = {
                'min': dates.min().isoformat() if not dates.empty else None,
                'max': dates.max().isoformat() if not dates.empty else None
            }
        
        # Categorical fields
        categorical_fields = [
            'CLASS', 'FORM_TYPE', 'VP_REGION_NAME', 'DF_MARKET_NAME', 
            'LOCATION_NAME', 'TMO_NAME', 'BRAND_NAME', 'PRODUCT_CATEGORY_NAME',
            'PMI_PRODUCT_NAME', 'BRAND_NAME_FROM', 'BRAND_NAME_TO'
        ]
        
        for field in categorical_fields:
            if field in df.columns:
                unique_values = df[field].dropna().unique().tolist()
                metadata[field] = sorted([str(v) for v in unique_values if str(v) != 'nan'])
        
        return metadata
        
    except Exception as e:
        logger.error(f"Error getting metadata: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/filter-data")
async def filter_data(filter_request: FilterRequest):
    """Filter data based on criteria"""
    try:
        df = data_cache.get('raw_data')
        if df is None or df.empty:
            df = load_csv_data()
        
        if df.empty:
            raise HTTPException(status_code=500, detail="No data available")
        
        filtered_df = df.copy()
        
        # Date range filter
        if filter_request.date_range and len(filter_request.date_range) == 2:
            start_date = pd.to_datetime(filter_request.date_range[0])
            end_date = pd.to_datetime(filter_request.date_range[1])
            if 'SUBMISSION_DATETIME' in filtered_df.columns:
                filtered_df = filtered_df[
                    (pd.to_datetime(filtered_df['SUBMISSION_DATETIME']) >= start_date) &
                    (pd.to_datetime(filtered_df['SUBMISSION_DATETIME']) <= end_date)
                ]
        
        # Categorical filters
        filter_mappings = {
            'feedback_class': 'CLASS',
            'region': 'VP_REGION_NAME',
            'market': 'DF_MARKET_NAME',
            'location': 'LOCATION_NAME',
            'tmo': 'TMO_NAME',
            'brand': 'BRAND_NAME',
            'product_category': 'PRODUCT_CATEGORY_NAME',
            'pmi_product': 'PMI_PRODUCT_NAME',
            'switch_from': 'BRAND_NAME_FROM',
            'switch_to': 'BRAND_NAME_TO'
        }
        
        for filter_key, column_name in filter_mappings.items():
            filter_values = getattr(filter_request, filter_key, [])
            if filter_values and column_name in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[column_name].isin(filter_values)]
        
        # Form type filter
        if filter_request.form_type != "All" and 'FORM_TYPE' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['FORM_TYPE'] == filter_request.form_type]
        
        # Add row IDs
        filtered_df = filtered_df.reset_index(drop=True)
        filtered_df['row_ID'] = filtered_df.index
        
        result = {
            'data': filtered_df.to_dict('records'),
            'count': len(filtered_df),
            'preview': filtered_df.head(10).to_dict('records') if len(filtered_df) > 0 else []
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error filtering data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def create_xml_data(df):
    """Convert DataFrame to XML format for report generation"""
    try:
        # Ensure required columns exist
        required_columns = [
            "FIELD_INTELLIGENCE_TRANSLATED", "SUBMISSION_DATETIME", "CLASS",
            "LOCATION_NAME", "FORM_TYPE", "PRODUCT_CATEGORY_NAME",
            "BRAND_NAME_FROM", "BRAND_NAME_TO", "PMI_PRODUCT_NAME", "TMO_NAME", "row_ID"
        ]
        
        # Create root element
        root = ET.Element("Entries")
        
        for _, row in df.iterrows():
            entry = ET.SubElement(root, "Entry")
            
            ET.SubElement(entry, "Row_ID").text = str(row.get('row_ID', ''))
            ET.SubElement(entry, "Class").text = str(row.get('CLASS', ''))
            ET.SubElement(entry, "Field_Intelligence_Translated").text = str(row.get('FIELD_INTELLIGENCE_TRANSLATED', ''))
            ET.SubElement(entry, "Submission_Date").text = str(row.get('SUBMISSION_DATETIME', ''))
            
            if pd.notna(row.get('LOCATION_NAME')):
                ET.SubElement(entry, "Location").text = str(row['LOCATION_NAME'])
            
            form_type = ET.SubElement(entry, "Form_Type")
            form_type.text = str(row.get('FORM_TYPE', ''))
            
            form_type_details = ET.SubElement(entry, "Form_Type_Details")
            detail_fields = ["PRODUCT_CATEGORY_NAME", "BRAND_NAME_FROM", "BRAND_NAME_TO", "PMI_PRODUCT_NAME", "TMO_NAME"]
            
            for field in detail_fields:
                if pd.notna(row.get(field)):
                    ET.SubElement(form_type_details, field.lower()).text = str(row[field])
        
        return ET.tostring(root, encoding="unicode", method="xml")
        
    except Exception as e:
        logger.error(f"Error creating XML data: {e}")
        return "<Entries></Entries>"

@app.post("/api/generate-report")
async def generate_report(filter_request: FilterRequest):
    """Generate a report using Claude AI"""
    try:
        # First filter the data
        filter_response = await filter_data(filter_request)
        data = filter_response['data']
        
        if not data:
            raise HTTPException(status_code=400, detail="No data found for the given filters")
        
        # Convert to DataFrame for processing
        df = pd.DataFrame(data)
        
        # Create XML data for the prompt
        xml_data = create_xml_data(df)
        
        # Create the prompt
        prompt = f"""Act as a data analyst specializing in employee feedback and customer request management. Your objective is to create a comprehensive report that evaluates the current state of global duty-free operations, leveraging structured data from employee feedback and customer-related comments. Your task is to **present granular, concrete data points** without overly summarizing or generalizing the information. Don't give Impact analysis or Recommendations unless specifically asked for.

### <task overview>
You are tasked to analyze the provided data and generate a detailed report that addresses the following key aspects:

#### **Top 5 Global Operational Challenges**
- Identify and list the **top issues** affecting global duty-free operations. Provide the issues in **specific, concrete terms**, highlighting granular data points.
- Present direct data points (e.g., specific SKUs, prices, promotions, or durations of out-of-stock situations).
- Avoid summarizing; instead, provide precise observations drawn from the dataset.
- Categorize these issues into specific areas (e.g., inventory management, service quality, customer behavior).
- Support findings with **exact examples from the data**, citing relevant Field Intelligence and Row_ID values (as many as possible). Be very detailed.

### **<instructions>**
The report must include **Specific Examples**: For each problem, provide exact product names, SKUs, promotional details, or other concrete data points.

Don't add any descriptions. No impact analysis, no recommendations.
Support insights with multiple examples (as many as possible, minimum 5), citing their Row_ID in this format:
[Row_ID:row_id]
Example Citation Format:
- [Row_ID:80]

#### <output format>:
The output should be in XML format:
<response>
    <report>
        Complete report as a single text block with full structure and formatting.
    </report>
    <examplesID>
        A list of the Row_IDs corresponding to the specific examples cited in the report, in the order they appear.
    </examplesID>
</response>
</output format>
</instructions>

<formatting instructions>:
- List specific examples for each insight, such as SKUs, promotions, or exact timelines.
- Emphasize key details like product names, exact prices, or promotion terms. Avoid vague phrases. What we don't want: "Some products were out of stock.", "Many customers complained about pricing.". What we want: "Product X (SKU: 12345) was out of stock for 3 days.", "Customers reported a 20% price increase for Brand Y in the last month."
- Include tables with product names and details, if applicable.
</formatting instructions>

Step-by-Step Process:
1. Analyze the dataset to identify the top problems globally with precise examples.
2. Segment the data by location to highlight regional nuances.
3. Extract granular information, such as specific product details, pricing, or stock durations.
4. Provide exact Field Intelligence examples and Row_IDs.

Use this data to generate the report:
<data>
{xml_data}
</data>

Let's think step by step."""

        # Call Claude API
        response = claude_client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=4096,
            temperature=0.1,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        
        # Extract report from response
        content = response.content[0].text
        report = extract_report_from_xml(content)
        examples_id = extract_examples_from_xml(content)
        
        return {
            "report": report,
            "examples_id": examples_id,
            "raw_response": content,
            "data_count": len(data)
        }
        
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def extract_report_from_xml(content: str) -> str:
    """Extract report content from XML response"""
    try:
        start = content.find("<response>")
        if start == -1:
            return content
        
        xml_content = content[start:]
        
        # Escape problematic characters in <report> content
        xml_content = re.sub(
            r"<report>(.*?)</report>",
            lambda m: f"<report><![CDATA[{m.group(1).strip()}]]></report>",
            xml_content,
            flags=re.DOTALL,
        )
        
        root = ET.fromstring(xml_content)
        report = root.find("report")
        return report.text.strip() if report is not None else content
        
    except Exception as e:
        logger.error(f"Error extracting report from XML: {e}")
        return content

def extract_examples_from_xml(content: str) -> List[str]:
    """Extract examples ID from XML response"""
    try:
        pattern = r"<examplesID>(.*?)</examplesID>"
        matches = re.findall(pattern, content, flags=re.DOTALL)
        
        if not matches:
            return []
        
        results = []
        for match in matches:
            items = re.split(r"[,\n]", match)
            results.extend(item.strip() for item in items if item.strip())
        
        return results
        
    except Exception as e:
        logger.error(f"Error extracting examples from XML: {e}")
        return []

@app.post("/api/chat")
async def chat_with_claude(chat_request: ChatRequest):
    """Handle follow-up chat with Claude"""
    try:
        # Convert messages to Claude format
        messages = []
        for msg in chat_request.messages:
            messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        # Call Claude API with system prompt
        response = claude_client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=4096,
            temperature=0.1,
            system=chat_request.system_prompt,
            messages=messages
        )
        
        return {
            "response": response.content[0].text,
            "role": "assistant"
        }
        
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/charts-data")
async def get_charts_data(filter_request: FilterRequest = None):
    """Get data for charts visualization"""
    try:
        if filter_request:
            filter_response = await filter_data(filter_request)
            data = filter_response['data']
        else:
            df = data_cache.get('raw_data', pd.DataFrame())
            data = df.to_dict('records') if not df.empty else []
        
        if not data:
            return {"charts": []}
        
        df = pd.DataFrame(data)
        
        charts_data = {}
        
        # Class distribution
        if 'CLASS' in df.columns:
            class_counts = df['CLASS'].value_counts().to_dict()
            charts_data['class_distribution'] = {
                'labels': list(class_counts.keys()),
                'values': list(class_counts.values())
            }
        
        # Monthly trends
        if 'SUBMISSION_DATETIME' in df.columns:
            df['SUBMISSION_DATETIME'] = pd.to_datetime(df['SUBMISSION_DATETIME'])
            monthly_counts = df.groupby(df['SUBMISSION_DATETIME'].dt.to_period('M')).size()
            charts_data['monthly_trends'] = {
                'labels': [str(period) for period in monthly_counts.index],
                'values': monthly_counts.values.tolist()
            }
        
        # Location distribution
        if 'LOCATION_NAME' in df.columns:
            location_counts = df['LOCATION_NAME'].value_counts().head(10).to_dict()
            charts_data['location_distribution'] = {
                'labels': list(location_counts.keys()),
                'values': list(location_counts.values())
            }
        
        return {"charts": charts_data}
        
    except Exception as e:
        logger.error(f"Error getting charts data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)