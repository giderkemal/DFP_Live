import os
import json
import boto3
import pandas as pd
import streamlit as st
import snowflake.connector
from http import HTTPStatus
from dotenv import load_dotenv

from snowflake.connector.pandas_tools import write_pandas

load_dotenv()

env = os.getenv('FDF_ENV', 'dev')

secret_manager_access_key = os.getenv('SECRET_MANAGER_ACCESS_KEY')
secret_manager_secret_key = os.getenv('SECRET_MANAGER_SECRET_KEY')

db_env = f'{env}s' if env == 'qa' else env
db_env = db_env.upper()


class SnowflakeConnectionService:
    @staticmethod
    def __get_service_account_credentials() -> dict:
        client = boto3.client(
            'secretsmanager',
            aws_access_key_id='AKIA5OWBNOABRF5KP7NX',
            aws_secret_access_key='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',
            region_name='eu-west-1',
        )

        response = client.get_secret_value(SecretId=f'vapafpdf-{env}-Snowflake_FDF_Password')

        secret = response['SecretString']
        secret_dict = json.loads(secret)

        credentials = {
            'service_account_password': secret_dict['password'],
            'service_account_username': secret_dict['username'],
        }

        return credentials

    @staticmethod
    def connect_to_snowflake(credentials):
        return snowflake.connector.connect(
            user=credentials['service_account_username'],
            password=credentials['service_account_password'],
            account='pl47603.eu-west-1',
            warehouse=f'WH_{db_env}_ETL',
            database=f'DB_FDF_{db_env}',
            schema='STAGING'
        )

    @staticmethod
    def execute_query(query):
        credentials = SnowflakeConnectionService.__get_service_account_credentials()

        with SnowflakeConnectionService.connect_to_snowflake(credentials) as conn:
            try:
                print('Executing query:')
                print(query)
                conn.cursor().execute(query)
                print('Query executed successfully.')
            except Exception as e:
                print(f'Failed to execute Snowflake query.')
                raise e

    @staticmethod
    def fetch_query_result(query):
        credentials = SnowflakeConnectionService.__get_service_account_credentials()

        with SnowflakeConnectionService.connect_to_snowflake(credentials) as conn:
            try:
                df = pd.DataFrame(conn.cursor().execute(query).fetch_pandas_all())
                df.reset_index(drop=True, inplace=True)
            except Exception as e:
                print(f'Failed to fetch data from Snowflake.')
                raise e

        return df

    @staticmethod
    def save_to_snowflake(df: pd.DataFrame, table_name: str, schema: str, overwrite: bool):
        try:
            credentials = SnowflakeConnectionService.__get_service_account_credentials()

            with SnowflakeConnectionService.connect_to_snowflake(credentials) as conn:
                success, chunks, rows, _ = write_pandas(
                    conn=conn,
                    df=df,
                    table_name=table_name,
                    schema=schema,
                    database=f'DB_FDF_{db_env}',
                    chunk_size=10000,
                    overwrite=overwrite,
                )

            message = 'Data is saved to Snowflake successfully'
            status_code = HTTPStatus.OK

        except Exception as e:
            message = 'Failed to save data to Snowflake'
            raise e

        return status_code, message

    @staticmethod
    @st.cache_data(ttl=86400)
    def get_raw_base_matrix():
        if env in ('dev', 'qa'):
            source_table = 'information_mart.df_passport_final'
        else:
            source_table = 'cs_commrcl.dte_fact_df_passport'

        query = f'''
        select
        cast(submission_datetime as date) as SUBMISSION_DATETIME,
        form_type as FORM_TYPE,
        location_name as LOCATION_NAME,
        brand_name_from as BRAND_NAME_FROM,
        brand_name_to as BRAND_NAME_TO,
        product_category_name as PRODUCT_CATEGORY_NAME,
        pmi_product_name as PMI_PRODUCT_NAME,
        tmo_name as TMO_NAME,
        brand_name as BRAND_NAME,
        field_intelligence as FIELD_INTELLIGENCE,
        field_intelligence_translated as FIELD_INTELLIGENCE_TRANSLATED,
        photo_link as PHOTO_LINK,
        pov_id as POV_ID,
        class as CLASS,
        df_market_name as DF_MARKET_NAME,
        vp_region_name as VP_REGION_NAME

        from {source_table}
        '''
        return SnowflakeConnectionService.fetch_query_result(query)
