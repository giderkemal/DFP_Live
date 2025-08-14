import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="DFP Data Interactive Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .filter-section {
        background-color: #fafafa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache(allow_output_mutation=True)
def load_data():
    """Load and preprocess the DFP data"""
    try:
        df = pd.read_csv('DFP_DATA.csv', low_memory=False)
        
        # Clean and preprocess data
        df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y', errors='coerce')
        df['Transaction Points'] = pd.to_numeric(df['Transaction Points'], errors='coerce')
        
        # Remove rows that are just filters or metadata
        df = df[df['Form Type'].notna() & (df['Form Type'] != 'Applied filters:')]
        
        # Clean text fields
        text_columns = ['Field Intelligence', 'Name', 'Username']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).replace('nan', '')
        
        # Extract year and month for analysis
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Year_Month'] = df['Date'].dt.to_period('M')
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def create_overview_metrics(df):
    """Create overview metrics cards"""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Records", f"{len(df):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Unique Locations", df['Location'].nunique())
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Countries/Markets", df['Market'].nunique())
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Date Range", f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col5:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        avg_points = df['Transaction Points'].mean() if df['Transaction Points'].notna().sum() > 0 else 0
        st.metric("Avg Transaction Points", f"{avg_points:.1f}")
        st.markdown('</div>', unsafe_allow_html=True)

def create_geographic_visualizations(df):
    """Create geographic analysis charts"""
    st.subheader("🌍 Geographic Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Regional distribution
        region_counts = df['Region'].value_counts()
        fig_region = px.pie(
            values=region_counts.values,
            names=region_counts.index,
            title="Distribution by Region",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_region.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_region)
    
    with col2:
        # Top markets by activity
        market_counts = df['Market'].value_counts().head(10)
        fig_market = px.bar(
            x=market_counts.values,
            y=market_counts.index,
            orientation='h',
            title="Top 10 Markets by Activity",
            labels={'x': 'Number of Records', 'y': 'Market'}
        )
        fig_market.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_market)

def create_temporal_analysis(df):
    """Create time-based analysis"""
    st.subheader("📅 Temporal Analysis")
    
    # Time series of submissions
    daily_counts = df.groupby(df['Date'].dt.date).size().reset_index()
    daily_counts.columns = ['Date', 'Count']
    
    fig_time = px.line(
        daily_counts,
        x='Date',
        y='Count',
        title="Daily Submission Trends",
        markers=True
    )
    fig_time.update_layout(xaxis_title="Date", yaxis_title="Number of Submissions")
    st.plotly_chart(fig_time)
    
    # Monthly heatmap
    if len(df) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            # Monthly activity by form type
            monthly_form = df.groupby(['Year_Month', 'Form Type']).size().reset_index()
            monthly_form.columns = ['Year_Month', 'Form_Type', 'Count']
            monthly_form['Year_Month'] = monthly_form['Year_Month'].astype(str)
            
            fig_monthly = px.bar(
                monthly_form,
                x='Year_Month',
                y='Count',
                color='Form_Type',
                title="Monthly Activity by Form Type",
                barmode='stack'
            )
            fig_monthly.update_layout(xaxis={'tickangle': 45})
            st.plotly_chart(fig_monthly)
        
        with col2:
            # Day of week analysis
            df['Day_of_Week'] = df['Date'].dt.day_name()
            dow_counts = df['Day_of_Week'].value_counts()
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            dow_counts = dow_counts.reindex([day for day in day_order if day in dow_counts.index])
            
            fig_dow = px.bar(
                x=dow_counts.index,
                y=dow_counts.values,
                title="Activity by Day of Week",
                labels={'x': 'Day of Week', 'y': 'Number of Submissions'}
            )
            st.plotly_chart(fig_dow)

def create_form_type_analysis(df):
    """Analyze form types and categories"""
    st.subheader("📋 Form Type & Category Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Form type distribution
        form_counts = df['Form Type'].value_counts()
        fig_forms = px.bar(
            x=form_counts.index,
            y=form_counts.values,
            title="Distribution by Form Type",
            labels={'x': 'Form Type', 'y': 'Count'}
        )
        fig_forms.update_layout(xaxis={'tickangle': 45})
        st.plotly_chart(fig_forms)
    
    with col2:
        # TMO analysis (excluding NaN)
        tmo_data = df[df['TMO'].notna() & (df['TMO'] != '')]
        if len(tmo_data) > 0:
            tmo_counts = tmo_data['TMO'].value_counts().head(10)
            fig_tmo = px.pie(
                values=tmo_counts.values,
                names=tmo_counts.index,
                title="Top TMO Distribution",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig_tmo)

def create_brand_analysis(df):
    """Analyze brand mentions and insights"""
    st.subheader("🏷️ Brand & Intelligence Analysis")
    
    # Filter data with brand information
    brand_data = df[df['Brand Name'].notna() & (df['Brand Name'] != '')]
    
    if len(brand_data) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            # Top brands mentioned
            brand_counts = brand_data['Brand Name'].value_counts().head(15)
            fig_brands = px.bar(
                x=brand_counts.values,
                y=brand_counts.index,
                orientation='h',
                title="Top 15 Brands Mentioned",
                labels={'x': 'Mentions', 'y': 'Brand'}
            )
            fig_brands.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_brands)
        
        with col2:
            # Brand mentions by region
            brand_region = brand_data.groupby(['Region', 'Brand Name']).size().reset_index()
            brand_region.columns = ['Region', 'Brand', 'Count']
            top_brands = brand_data['Brand Name'].value_counts().head(5).index
            brand_region_top = brand_region[brand_region['Brand'].isin(top_brands)]
            
            fig_brand_region = px.bar(
                brand_region_top,
                x='Region',
                y='Count',
                color='Brand',
                title="Top 5 Brands by Region",
                barmode='stack'
            )
            fig_brand_region.update_layout(xaxis={'tickangle': 45})
            st.plotly_chart(fig_brand_region, use_container_width=True)
    
    # Field Intelligence text analysis
    intel_data = df[df['Field Intelligence'].notna() & (df['Field Intelligence'] != '') & (df['Field Intelligence'] != 'nan')]
    if len(intel_data) > 0:
        st.subheader("💡 Field Intelligence Insights")
        
        # Show recent intelligence samples
        recent_intel = intel_data.nlargest(10, 'Date')[['Date', 'Region', 'Market', 'Form Type', 'Field Intelligence']]
        st.dataframe(recent_intel)

def create_user_activity_analysis(df):
    """Analyze user activity and contributions"""
    st.subheader("👥 User Activity Analysis")
    
    user_data = df[df['Username'].notna() & (df['Username'] != '') & (df['Username'] != 'nan')]
    
    if len(user_data) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            # Top contributors
            user_counts = user_data['Username'].value_counts().head(10)
            fig_users = px.bar(
                x=user_counts.values,
                y=user_counts.index,
                orientation='h',
                title="Top 10 Contributors",
                labels={'x': 'Submissions', 'y': 'Username'}
            )
            fig_users.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_users, use_container_width=True)
        
        with col2:
            # User activity by region
            user_region = user_data.groupby('Region')['Username'].nunique().reset_index()
            user_region.columns = ['Region', 'Unique_Users']
            
            fig_user_region = px.bar(
                user_region,
                x='Region',
                y='Unique_Users',
                title="Unique Users by Region",
                labels={'y': 'Number of Unique Users'}
            )
            fig_user_region.update_layout(xaxis={'tickangle': 45})
            st.plotly_chart(fig_user_region, use_container_width=True)

def create_filters_sidebar(df):
    """Create interactive filters in sidebar"""
    st.sidebar.markdown('<div class="filter-section">', unsafe_allow_html=True)
    st.sidebar.header("🔍 Filters")
    
    # Date range filter
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Region filter
    regions = ['All'] + sorted(df['Region'].dropna().unique().tolist())
    selected_region = st.sidebar.selectbox("Select Region", regions)
    
    # Form type filter
    form_types = ['All'] + sorted(df['Form Type'].dropna().unique().tolist())
    selected_form_type = st.sidebar.selectbox("Select Form Type", form_types)
    
    # Market filter (only show markets from selected region)
    if selected_region != 'All':
        markets = ['All'] + sorted(df[df['Region'] == selected_region]['Market'].dropna().unique().tolist())
    else:
        markets = ['All'] + sorted(df['Market'].dropna().unique().tolist())
    selected_market = st.sidebar.selectbox("Select Market", markets)
    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    return date_range, selected_region, selected_form_type, selected_market

def apply_filters(df, date_range, region, form_type, market):
    """Apply selected filters to dataframe"""
    filtered_df = df.copy()
    
    # Date filter
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df['Date'].dt.date >= start_date) & 
            (filtered_df['Date'].dt.date <= end_date)
        ]
    
    # Region filter
    if region != 'All':
        filtered_df = filtered_df[filtered_df['Region'] == region]
    
    # Form type filter
    if form_type != 'All':
        filtered_df = filtered_df[filtered_df['Form Type'] == form_type]
    
    # Market filter
    if market != 'All':
        filtered_df = filtered_df[filtered_df['Market'] == market]
    
    return filtered_df

def main():
    """Main dashboard function"""
    st.markdown('<h1 class="main-header">📊 DFP Data Interactive Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
    
    if df.empty:
        st.error("No data available. Please check the DFP_DATA.csv file.")
        return
    
    # Create filters
    date_range, selected_region, selected_form_type, selected_market = create_filters_sidebar(df)
    
    # Apply filters
    filtered_df = apply_filters(df, date_range, selected_region, selected_form_type, selected_market)
    
    # Show filter status
    if len(filtered_df) != len(df):
        st.info(f"Showing {len(filtered_df):,} of {len(df):,} records based on filters")
    
    # Overview metrics
    create_overview_metrics(filtered_df)
    st.markdown("---")
    
    # Create tabs for different analysis sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌍 Geographic", "📅 Temporal", "📋 Form Types", 
        "🏷️ Brands & Intelligence", "👥 User Activity"
    ])
    
    with tab1:
        create_geographic_visualizations(filtered_df)
    
    with tab2:
        create_temporal_analysis(filtered_df)
    
    with tab3:
        create_form_type_analysis(filtered_df)
    
    with tab4:
        create_brand_analysis(filtered_df)
    
    with tab5:
        create_user_activity_analysis(filtered_df)
    
    # Raw data section
    with st.expander("📋 Raw Data View", expanded=False):
        st.dataframe(filtered_df)
        
        # Download filtered data
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv,
            file_name=f"dfp_filtered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main() 