import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import uuid

# Create sample data that matches the expected structure
def create_sample_data():
    np.random.seed(42)  # For reproducible data
    
    # Define sample values
    regions = ['North America', 'Europe', 'Asia Pacific', 'Latin America', 'Africa']
    markets = ['USA', 'Germany', 'Japan', 'Brazil', 'South Africa', 'UK', 'France', 'Australia', 'Canada', 'Italy']
    locations = ['Location_A', 'Location_B', 'Location_C', 'Location_D', 'Location_E', 'Location_F', 'Location_G', 'Location_H']
    form_types = ['CONSUMER_FEEDBACK', 'BRAND_SOURCING', 'CROSS_CATEGORY', 'TOBACCO_CATEGORY', 'INFRA_MAINTENANCE']
    classes = ['Pricing', 'Product Quality', 'Availability', 'Service', 'Competition', 'Regulatory', 'Consumer Behavior']
    tmos = ['TMO_Alpha', 'TMO_Beta', 'TMO_Gamma', 'TMO_Delta', 'TMO_Epsilon']
    product_categories = ['Traditional', 'Heated Tobacco', 'E-Cigarettes', 'Accessories', 'Nicotine Pouches']
    brands_from = ['Brand_X', 'Brand_Y', 'Brand_Z', 'Brand_A', 'Brand_B', 'Competitor_1', 'Competitor_2']
    brands_to = ['PMI_Brand_1', 'PMI_Brand_2', 'PMI_Brand_3', 'IQOS', 'Marlboro']
    brand_names = brands_from + brands_to
    pmi_products = ['IQOS', 'HEETS', 'Marlboro', 'Parliament', 'Philip Morris']
    
    # Sample field intelligence texts
    sample_texts = [
        "Customer reported price increase concern affecting purchase decisions",
        "Positive feedback on new product launch in the region",
        "Stock shortage observed for premium product line",
        "Competitor promotion impacting market share",
        "Service quality issues reported at retail locations",
        "Strong consumer interest in new heated tobacco category",
        "Regulatory changes affecting product availability",
        "Customer satisfaction high with recent product improvements",
        "Market expansion opportunities identified in urban areas",
        "Price sensitivity observed among younger demographics",
        "Product availability issues during peak season",
        "Positive brand perception following marketing campaign",
        "Distribution challenges in remote locations",
        "Consumer switching behavior analysis shows retention improvement",
        "Retail partner feedback on product performance",
        "Competitive pricing pressure in premium segment",
        "New product category showing strong adoption",
        "Service training needs identified for retail staff",
        "Market research indicates growth opportunity",
        "Consumer preference shift towards premium products"
    ]
    
    # Generate sample data
    num_records = 5000  # Generate 5000 sample records
    data = []
    
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    for i in range(num_records):
        # Random date within range
        random_date = start_date + timedelta(days=np.random.randint(0, (end_date - start_date).days))
        
        # Select form type first to determine available fields
        form_type = np.random.choice(form_types)
        
        record = {
            'SUBMISSION_DATETIME': random_date.strftime('%Y-%m-%d %H:%M:%S'),
            'FORM_TYPE': form_type,
            'VP_REGION_NAME': np.random.choice(regions),
            'DF_MARKET_NAME': np.random.choice(markets),
            'LOCATION_NAME': np.random.choice(locations),
            'CLASS': np.random.choice(classes),
            'FIELD_INTELLIGENCE_TRANSLATED': np.random.choice(sample_texts),
            'TMO_NAME': np.random.choice(tmos) if form_type in ['TOBACCO_CATEGORY', 'All'] else None,
            'PRODUCT_CATEGORY_NAME': np.random.choice(product_categories),
            'PMI_PRODUCT_NAME': np.random.choice(pmi_products) if form_type == 'CONSUMER_FEEDBACK' else None,
            'BRAND_NAME': np.random.choice(brand_names) if form_type in ['TOBACCO_CATEGORY', 'All'] else None,
            'BRAND_NAME_FROM': np.random.choice(brands_from) if form_type == 'BRAND_SOURCING' else None,
            'BRAND_NAME_TO': np.random.choice(brands_to) if form_type == 'BRAND_SOURCING' else None,
        }
        
        data.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv('/app/DFP_DATA.csv', index=False)
    print(f"Created sample data with {len(df)} records")
    print(f"Columns: {list(df.columns)}")
    print(f"Date range: {df['SUBMISSION_DATETIME'].min()} to {df['SUBMISSION_DATETIME'].max()}")
    
    return df

if __name__ == "__main__":
    create_sample_data()