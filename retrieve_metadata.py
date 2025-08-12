import pandas as pd
import streamlit as st

from snowflake_connection import SnowflakeConnectionService


class RetrieveMetadataService:
    @staticmethod
    @st.cache_data(ttl=86400)
    def get_unique_metadata_combinations():
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

        unique_combinations = SnowflakeConnectionService.fetch_query_result(query)

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
        }

        metadata_renamed = unique_combinations.rename(columns=rename_dict)

        return metadata_renamed

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