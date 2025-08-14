import pandas as pd
import streamlit as st
import logging

from snowflake_connection import SnowflakeConnectionService

# Configure logging
logger = logging.getLogger(__name__)

class RetrieveMetadataService:
    @staticmethod
    @st.cache_data(ttl=86400)
    def get_unique_metadata_combinations():
        """
        Enhanced metadata retrieval with fallback handling for Snowflake connection issues
        """
        query = """
            select distinct
            date(df.submission_datetime) as date,
            df.form_type,
            cls.class,
            df.location_name,
            df.brand_name_from,
            df.brand_name,
            df.brand_name_to,
            case when df.form_type = 'CONSUMER_FEEDBACK' then df.product_category_name
                else null 
                end as answer_category,
            case when df.form_type != 'CONSUMER_FEEDBACK' then df.product_category_name
                else null 
                end as product_category_name,
            df.pmi_product_name,
            df.tmo_name,
            df.df_market_name,
            df.vp_region_name

            from information_mart.df_passport_final df
            left join information_mart.df_passport_classification cls on df.task_code = cls.task_code
            """

        try:
            logger.info("Attempting to fetch metadata from Snowflake")
            unique_combinations = SnowflakeConnectionService.fetch_query_result(query)
            
            # Check if we got valid data
            if unique_combinations is None or unique_combinations.empty:
                logger.warning("Snowflake returned empty result, using fallback data")
                return RetrieveMetadataService._get_fallback_metadata()
            
            logger.info(f"Successfully retrieved {len(unique_combinations)} metadata combinations from Snowflake")
            
            # Apply renaming
            rename_dict = {
                "VP_REGION_NAME": "Region",
                "DF_MARKET_NAME": "Market",
                "LOCATION_NAME": "Location",
                "TMO_NAME": "TMO",
                "PRODUCT_CATEGORY_NAME": "Product Category",
                "PMI_PRODUCT_NAME": "PMI Product",
                "BRAND_NAME": "Brand",
                "BRAND_NAME_FROM": "Switch From",
                "BRAND_NAME_TO": "Switch To",
                "ANSWER_CATEGORY": "Answer Category",
                "CLASS": "Class",
                "FORM_TYPE": "Form Type",
                "DATE": "Date"
            }

            # Apply column renaming
            unique_combinations = unique_combinations.rename(columns=rename_dict)
            
            # Additional processing
            unique_combinations["Date"] = pd.to_datetime(unique_combinations["Date"])
            unique_combinations = unique_combinations.dropna(subset=["Date"])

            logger.info("Successfully processed metadata from Snowflake")
            return unique_combinations
            
        except Exception as e:
            logger.error(f"Failed to retrieve metadata from Snowflake: {e}")
            logger.info("Using fallback metadata for app functionality")
            
            # Show user-friendly message
            st.warning("⚠️ Unable to connect to Snowflake database. Using sample data for demonstration.")
            st.info("To enable full functionality, please check your Snowflake connection settings.")
            
            return RetrieveMetadataService._get_fallback_metadata()
    
    @staticmethod
    def _get_fallback_metadata():
        """
        Provide fallback metadata when Snowflake is unavailable
        """
        logger.info("Generating fallback metadata")
        
        # Create sample data structure that matches expected format
        fallback_data = {
            "Date": pd.date_range(start="2024-01-01", end="2024-12-31", freq="D"),
            "Form Type": ["CONSUMER_FEEDBACK", "BRAND_SOURCING", "CROSS_CATEGORY", "TOBACCO_CATEGORY"] * 92,
            "Class": ["Pricing", "Product Quality", "Availability", "Service", "Other"] * 73,
            "Location": ["Sample Location A", "Sample Location B", "Sample Location C"] * 122,
            "Switch From": ["Sample Brand X", "Sample Brand Y", "Sample Brand Z"] * 122,
            "Brand": ["Sample Brand 1", "Sample Brand 2", "Sample Brand 3"] * 122,
            "Switch To": ["Sample Target A", "Sample Target B", "Sample Target C"] * 122,
            "Answer Category": ["Sample Category 1", "Sample Category 2", "Sample Category 3"] * 122,
            "Product Category": ["Sample Product A", "Sample Product B", "Sample Product C"] * 122,
            "PMI Product": ["Sample PMI Product 1", "Sample PMI Product 2"] * 183,
            "TMO": ["Sample TMO 1", "Sample TMO 2", "Sample TMO 3"] * 122,
            "Market": ["Sample Market A", "Sample Market B"] * 183,
            "Region": ["Sample Region 1", "Sample Region 2", "Sample Region 3"] * 122
        }
        
        # Create DataFrame with consistent length
        max_length = 366  # One year of daily data
        
        for key in fallback_data:
            # Ensure all arrays have the same length
            while len(fallback_data[key]) < max_length:
                fallback_data[key].extend(fallback_data[key][:min(len(fallback_data[key]), max_length - len(fallback_data[key]))])
            fallback_data[key] = fallback_data[key][:max_length]
        
        fallback_df = pd.DataFrame(fallback_data)
        
        logger.info(f"Generated fallback metadata with {len(fallback_df)} rows")
        return fallback_df

    @staticmethod
    def filter_metadata(df: pd.DataFrame, scope: dict) -> dict:
        """
        Adjust metadata based on selected options in filters.

        :param df: limited metadata DataFrame containing unique scope combinations.
        :param scope: scope selected by user.
        :return: dictionary with all available options per each scope field.
        """

        query_list = [
            f'{key} in {tuple(value)}'
            for key, value in scope.items() if value
        ]

        if query_list:
            filtered_df = df.query(' & '.join(query_list))

        else:
            filtered_df = df

        available_columns = [
            col for col in filtered_df.columns if col not in scope.keys()
        ]

        available_options = {
            col: filtered_df[col].unique().tolist()
            for col in available_columns
        }

        for key in scope.keys():
            query = ' & '.join([q for q in query_list if (key not in q)])

            filtered_df = df.query(query) if query else df

            available_options[key] = filtered_df[key].unique().tolist()

        return available_options