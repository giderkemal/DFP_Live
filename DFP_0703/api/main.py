from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Optional, List, Dict, Any
import json
from pathlib import Path

app = FastAPI(title="DFP Analytics API", version="1.0.0")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global data storage
df_cache = None

def load_data():
    """Load and preprocess the DFP data"""
    global df_cache
    if df_cache is not None:
        return df_cache
    
    try:
        # Load CSV file
        csv_path = Path("../DFP_DATA.csv")
        if not csv_path.exists():
            csv_path = Path("DFP_DATA.csv")
        
        df = pd.read_csv(csv_path, low_memory=False)
        
        # Clean and preprocess data
        df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y', errors='coerce')
        df['Transaction Points'] = pd.to_numeric(df['Transaction Points'], errors='coerce')
        
        # Remove rows that are just filters or metadata
        df = df[df['Form Type'].notna() & (df['Form Type'] != 'Applied filters:')]
        
        # Clean text fields
        text_columns = ['Field Intelligence', 'Name', 'Username', 'TMO', 'Brand Name']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).replace('nan', '').replace('None', '')
        
        # Extract temporal features
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['Weekday'] = df['Date'].dt.day_name()
        df['Week_Number'] = df['Date'].dt.isocalendar().week
        
        # Create ID column for frontend
        df['id'] = range(len(df))
        
        df_cache = df
        return df
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading data: {str(e)}")

@app.get("/")
async def root():
    return {"message": "DFP Analytics API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/data/summary")
async def get_data_summary():
    """Get basic summary statistics of the dataset"""
    df = load_data()
    
    summary = {
        "total_records": len(df),
        "date_range": {
            "start": df['Date'].min().isoformat() if df['Date'].min() else None,
            "end": df['Date'].max().isoformat() if df['Date'].max() else None
        },
        "unique_counts": {
            "regions": df['Region'].nunique(),
            "markets": df['Market'].nunique(), 
            "locations": df['Location'].nunique(),
            "form_types": df['Form Type'].nunique(),
            "users": df['Username'].nunique()
        },
        "columns": list(df.columns)
    }
    
    return summary

@app.get("/data/filters")
async def get_filter_options():
    """Get all available filter options for the frontend"""
    df = load_data()
    
    return {
        "regions": sorted(df['Region'].dropna().unique().tolist()),
        "markets": sorted(df['Market'].dropna().unique().tolist()),
        "form_types": sorted(df['Form Type'].dropna().unique().tolist()),
        "tmos": sorted(df[df['TMO'].notna() & (df['TMO'] != '')]['TMO'].unique().tolist()),
        "years": sorted(df['Year'].dropna().unique().astype(int).tolist()),
        "months": list(range(1, 13))
    }

@app.get("/data/records")
async def get_filtered_data(
    page: int = Query(1, ge=1),
    limit: int = Query(100, ge=1, le=1000),
    region: Optional[str] = None,
    market: Optional[str] = None,
    form_type: Optional[str] = None,
    tmo: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    search: Optional[str] = None
):
    """Get filtered and paginated data records"""
    df = load_data()
    
    # Apply filters
    filtered_df = df.copy()
    
    if region:
        filtered_df = filtered_df[filtered_df['Region'] == region]
    
    if market:
        filtered_df = filtered_df[filtered_df['Market'] == market]
    
    if form_type:
        filtered_df = filtered_df[filtered_df['Form Type'] == form_type]
    
    if tmo:
        filtered_df = filtered_df[filtered_df['TMO'] == tmo]
    
    if start_date:
        start = pd.to_datetime(start_date)
        filtered_df = filtered_df[filtered_df['Date'] >= start]
    
    if end_date:
        end = pd.to_datetime(end_date)
        filtered_df = filtered_df[filtered_df['Date'] <= end]
    
    if search:
        # Search across multiple text columns
        search_cols = ['Field Intelligence', 'Name', 'Username', 'Brand Name']
        search_mask = False
        for col in search_cols:
            if col in filtered_df.columns:
                search_mask |= filtered_df[col].str.contains(search, case=False, na=False)
        filtered_df = filtered_df[search_mask]
    
    # Calculate pagination
    total_records = len(filtered_df)
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    
    # Get paginated results
    paginated_df = filtered_df.iloc[start_idx:end_idx]
    
    # Convert to records format
    records = []
    for _, row in paginated_df.iterrows():
        record = row.to_dict()
        # Convert timestamps to ISO format
        if pd.notna(record['Date']):
            record['Date'] = record['Date'].isoformat()
        # Handle NaN values
        for key, value in record.items():
            if pd.isna(value):
                record[key] = None
        records.append(record)
    
    return {
        "records": records,
        "pagination": {
            "page": page,
            "limit": limit,
            "total_records": total_records,
            "total_pages": (total_records + limit - 1) // limit
        }
    }

@app.get("/analytics/geographic")
async def get_geographic_analytics():
    """Get geographic distribution analytics"""
    df = load_data()
    
    # Regional distribution
    region_counts = df['Region'].value_counts().to_dict()
    
    # Market distribution
    market_counts = df['Market'].value_counts().head(15).to_dict()
    
    # Location distribution by region
    location_by_region = df.groupby(['Region', 'Market']).size().reset_index(name='count')
    location_data = []
    for _, row in location_by_region.iterrows():
        location_data.append({
            "region": row['Region'],
            "market": row['Market'],
            "count": int(row['count'])
        })
    
    return {
        "region_distribution": region_counts,
        "market_distribution": market_counts,
        "location_hierarchy": location_data
    }

@app.get("/analytics/temporal")
async def get_temporal_analytics():
    """Get time-based analytics"""
    df = load_data()
    
    # Daily trends
    daily_counts = df.groupby(df['Date'].dt.date).size().reset_index()
    daily_counts.columns = ['date', 'count']
    daily_trends = []
    for _, row in daily_counts.iterrows():
        daily_trends.append({
            "date": row['date'].isoformat(),
            "count": int(row['count'])
        })
    
    # Monthly trends by form type
    monthly_form = df.groupby([df['Date'].dt.to_period('M'), 'Form Type']).size().reset_index()
    monthly_form.columns = ['month', 'form_type', 'count']
    monthly_data = []
    for _, row in monthly_form.iterrows():
        monthly_data.append({
            "month": str(row['month']),
            "form_type": row['form_type'],
            "count": int(row['count'])
        })
    
    # Weekday distribution
    weekday_counts = df['Weekday'].value_counts().to_dict()
    
    return {
        "daily_trends": daily_trends,
        "monthly_by_form_type": monthly_data,
        "weekday_distribution": weekday_counts
    }

@app.get("/analytics/brands")
async def get_brand_analytics():
    """Get brand and TMO analytics"""
    df = load_data()
    
    # Brand mentions (excluding empty/nan)
    brand_data = df[df['Brand Name'].notna() & (df['Brand Name'] != '') & (df['Brand Name'] != 'nan')]
    brand_counts = brand_data['Brand Name'].value_counts().head(20).to_dict()
    
    # TMO distribution
    tmo_data = df[df['TMO'].notna() & (df['TMO'] != '') & (df['TMO'] != 'nan')]
    tmo_counts = tmo_data['TMO'].value_counts().to_dict()
    
    # Brand mentions by region
    brand_region = brand_data.groupby(['Region', 'Brand Name']).size().reset_index(name='count')
    brand_region_data = []
    top_brands = brand_data['Brand Name'].value_counts().head(10).index
    for _, row in brand_region[brand_region['Brand Name'].isin(top_brands)].iterrows():
        brand_region_data.append({
            "region": row['Region'],
            "brand": row['Brand Name'],
            "count": int(row['count'])
        })
    
    return {
        "brand_mentions": brand_counts,
        "tmo_distribution": tmo_counts,
        "brand_by_region": brand_region_data
    }

@app.get("/analytics/forms")
async def get_form_analytics():
    """Get form type analytics"""
    df = load_data()
    
    # Form type distribution
    form_counts = df['Form Type'].value_counts().to_dict()
    
    # Form types by region
    form_region = df.groupby(['Region', 'Form Type']).size().reset_index(name='count')
    form_region_data = []
    for _, row in form_region.iterrows():
        form_region_data.append({
            "region": row['Region'],
            "form_type": row['Form Type'],
            "count": int(row['count'])
        })
    
    # Transaction points analysis
    transaction_stats = {
        "mean": float(df['Transaction Points'].mean()) if df['Transaction Points'].notna().sum() > 0 else 0,
        "median": float(df['Transaction Points'].median()) if df['Transaction Points'].notna().sum() > 0 else 0,
        "std": float(df['Transaction Points'].std()) if df['Transaction Points'].notna().sum() > 0 else 0,
        "total": float(df['Transaction Points'].sum()) if df['Transaction Points'].notna().sum() > 0 else 0
    }
    
    return {
        "form_distribution": form_counts,
        "form_by_region": form_region_data,
        "transaction_statistics": transaction_stats
    }

@app.get("/analytics/users")
async def get_user_analytics():
    """Get user activity analytics"""
    df = load_data()
    
    # User data
    user_data = df[df['Username'].notna() & (df['Username'] != '') & (df['Username'] != 'nan')]
    
    # Top contributors
    user_counts = user_data['Username'].value_counts().head(20).to_dict()
    
    # Users by region
    user_region = user_data.groupby('Region')['Username'].nunique().reset_index()
    user_region.columns = ['region', 'unique_users']
    user_region_data = []
    for _, row in user_region.iterrows():
        user_region_data.append({
            "region": row['region'],
            "unique_users": int(row['unique_users'])
        })
    
    # User activity over time
    user_temporal = user_data.groupby(user_data['Date'].dt.to_period('M'))['Username'].nunique().reset_index()
    user_temporal.columns = ['month', 'active_users']
    user_temporal_data = []
    for _, row in user_temporal.iterrows():
        user_temporal_data.append({
            "month": str(row['month']),
            "active_users": int(row['active_users'])
        })
    
    return {
        "top_contributors": user_counts,
        "users_by_region": user_region_data,
        "user_activity_timeline": user_temporal_data
    }

@app.get("/export/csv")
async def export_filtered_data(
    region: Optional[str] = None,
    market: Optional[str] = None,
    form_type: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """Export filtered data as CSV"""
    df = load_data()
    
    # Apply same filters as get_filtered_data
    filtered_df = df.copy()
    
    if region:
        filtered_df = filtered_df[filtered_df['Region'] == region]
    if market:
        filtered_df = filtered_df[filtered_df['Market'] == market]
    if form_type:
        filtered_df = filtered_df[filtered_df['Form Type'] == form_type]
    if start_date:
        start = pd.to_datetime(start_date)
        filtered_df = filtered_df[filtered_df['Date'] >= start]
    if end_date:
        end = pd.to_datetime(end_date)
        filtered_df = filtered_df[filtered_df['Date'] <= end]
    
    # Convert to CSV
    csv_data = filtered_df.to_csv(index=False)
    
    from fastapi.responses import Response
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=dfp_filtered_data.csv"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 