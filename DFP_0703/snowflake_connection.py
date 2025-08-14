import os
import json
import boto3
import pandas as pd
import streamlit as st
import snowflake.connector
from http import HTTPStatus
from dotenv import load_dotenv
import logging

from snowflake.connector.pandas_tools import write_pandas

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

env = os.getenv('FDF_ENV', 'dev')

secret_manager_access_key = os.getenv('SECRET_MANAGER_ACCESS_KEY')
secret_manager_secret_key = os.getenv('SECRET_MANAGER_SECRET_KEY')

db_env = f'{env}s' if env == 'qa' else env
db_env = db_env.upper()


class SnowflakeConnectionService:
    @staticmethod
    def __get_service_account_credentials() -> dict:
        """
        Enhanced credential retrieval that tries AWS Secrets Manager first (since it worked before),
        then falls back to local environment variables
        """
        # Strategy 1: Try AWS Secrets Manager first (since it worked previously)
        try:
            aws_credentials = SnowflakeConnectionService.__get_aws_secrets_credentials()
            if aws_credentials and aws_credentials.get('service_account_username'):
                logger.info("Using AWS Secrets Manager for Snowflake connection")
                return aws_credentials
        except Exception as e:
            logger.warning(f"Failed to get AWS Secrets Manager credentials: {e}")
        
        # Strategy 2: Fall back to local environment variables
        try:
            local_credentials = SnowflakeConnectionService.__get_local_credentials()
            if local_credentials:
                logger.info("Using local environment variables for Snowflake connection")
                return local_credentials
        except Exception as e:
            logger.error(f"Failed to get local credentials: {e}")
            
        # Strategy 3: Final fallback with basic environment setup
        logger.warning("All credential strategies failed, using basic fallback")
        return {
            'service_account_username': os.getenv('user', 'KGIDER@PMINTL.NET'),
            'service_account_password': ""
        }
    
    @staticmethod
    def __get_local_credentials() -> dict:
        """
        Try to get Snowflake credentials from local environment variables
        """
        # Get username from environment
        username = os.getenv('user') or os.getenv('SNOWFLAKE_USER')
        
        # Try different password environment variable names
        password = (
            os.getenv('SNOWFLAKE_PASSWORD') or 
            os.getenv('password') or 
            os.getenv('SNOWFLAKE_PASS') or
            os.getenv('SERVICE_ACCOUNT_PASSWORD')
        )
        
        # Check if we have minimum required credentials
        if username:
            if not password:
                # If no password found, we can try to prompt for it or use SSO
                logger.warning("No password found in environment variables")
                # For now, we'll try to continue without password (SSO/key-based auth)
                password = ""
            
            credentials = {
                'service_account_username': username,
                'service_account_password': password,
            }
            
            logger.info(f"Local credentials found for user: {username}")
            return credentials
        else:
            logger.warning("No username found in local environment variables")
            return None
    
    @staticmethod
    def __get_aws_secrets_credentials() -> dict:
        """
        Enhanced AWS Secrets Manager credential retrieval with flexible key handling
        """
        try:
            client = boto3.client(
                'secretsmanager',
                aws_access_key_id=secret_manager_access_key or 'AKIA5OWBNOABRF5KP7NX',
                aws_secret_access_key=secret_manager_secret_key or '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',
                region_name='eu-west-1',
            )

            response = client.get_secret_value(SecretId=f'vapafpdf-{env}-Snowflake_FDF_Password')

            secret = response['SecretString']
            secret_dict = json.loads(secret)
            
            logger.info(f"Retrieved secret with keys: {list(secret_dict.keys())}")

            # Enhanced key mapping - try different possible key names
            username = None
            password = None
            
            # Try different possible username keys
            username_keys = ['username', 'user', 'service_account_username', 'snowflake_user', 'account_user']
            for key in username_keys:
                if key in secret_dict:
                    username = secret_dict[key]
                    logger.info(f"Found username in key: {key}")
                    break
            
            # Try different possible password keys  
            password_keys = ['password', 'service_account_password', 'snowflake_password', 'account_password', 'pass']
            for key in password_keys:
                if key in secret_dict:
                    password = secret_dict[key]
                    logger.info(f"Found password in key: {key}")
                    break
            
            # Handle case where password might be stored differently or not needed
            if not password:
                logger.warning("No password found in secret - checking for alternative auth methods")
                # Check for key-based auth or other credentials
                alt_keys = ['private_key', 'key_pair', 'authenticator', 'token']
                for key in alt_keys:
                    if key in secret_dict:
                        logger.info(f"Found alternative auth method: {key}")
                        password = secret_dict[key]  # Use as password placeholder
                        break
                
                if not password:
                    logger.info("No password or alternative auth found, attempting passwordless auth")
                    password = ""  # Empty password for SSO/key-based auth

            if not username:
                # If no username found, fall back to environment variable
                username = os.getenv('user', 'KGIDER@PMINTL.NET')
                logger.warning(f"No username in secret, using fallback: {username}")

            credentials = {
                'service_account_password': password,
                'service_account_username': username,
            }

            logger.info(f"AWS Secrets Manager credentials configured for user: {username}")
            return credentials
            
        except Exception as e:
            logger.error(f"Failed to retrieve AWS Secrets Manager credentials: {e}")
            # Instead of failing completely, try to provide a reasonable fallback
            logger.info("Attempting fallback credentials from environment")
            
            fallback_username = os.getenv('user', 'KGIDER@PMINTL.NET')
            fallback_credentials = {
                'service_account_password': "",  # Empty for SSO/key-based auth
                'service_account_username': fallback_username,
            }
            
            logger.warning(f"Using fallback credentials for user: {fallback_username}")
            return fallback_credentials

    @staticmethod
    def connect_to_snowflake(credentials):
        """
        Enhanced connection method that uses environment variables for connection parameters
        """
        # Get connection parameters from environment variables with fallbacks
        account = os.getenv('account', 'pl47603.eu-west-1').replace('"', '')
        warehouse = os.getenv('SNOWFLAKE_WAREHOUSE', f'WH_{db_env}_ETL')
        database = os.getenv('SNOWFLAKE_DATABASE', f'DB_FDF_{db_env}')
        schema = os.getenv('SNOWFLAKE_SCHEMA', 'STAGING')
        role = os.getenv('role', '').replace('"', '')
        
        # Connection parameters
        connection_params = {
            'user': credentials['service_account_username'],
            'account': account,
            'warehouse': warehouse,
            'database': database,
            'schema': schema,
        }
        
        # Add password if available
        if credentials.get('service_account_password'):
            connection_params['password'] = credentials['service_account_password']
        
        # Add role if specified
        if role:
            connection_params['role'] = role
        
        logger.info(f"Connecting to Snowflake with account: {account}, warehouse: {warehouse}, database: {database}")
        
        try:
            return snowflake.connector.connect(**connection_params)
        except Exception as e:
            logger.error(f"Failed to connect to Snowflake: {e}")
            # Try alternative authentication methods
            if 'password' in connection_params:
                logger.info("Retrying connection without password (SSO/key-based auth)")
                connection_params_no_pwd = connection_params.copy()
                del connection_params_no_pwd['password']
                try:
                    return snowflake.connector.connect(**connection_params_no_pwd)
                except Exception as e2:
                    logger.error(f"Alternative connection method also failed: {e2}")
            raise e

    @staticmethod
    def execute_query(query):
        """Enhanced query execution with better error handling"""
        try:
            credentials = SnowflakeConnectionService.__get_service_account_credentials()

            with SnowflakeConnectionService.connect_to_snowflake(credentials) as conn:
                try:
                    logger.info('Executing Snowflake query')
                    conn.cursor().execute(query)
                    logger.info('Query executed successfully')
                except Exception as e:
                    logger.error(f'Failed to execute Snowflake query: {e}')
                    raise e
        except Exception as e:
            logger.error(f'Failed to establish Snowflake connection for query execution: {e}')
            raise e

    @staticmethod
    def fetch_query_result(query):
        """Enhanced query result fetching with better error handling"""
        try:
            credentials = SnowflakeConnectionService.__get_service_account_credentials()

            with SnowflakeConnectionService.connect_to_snowflake(credentials) as conn:
                try:
                    logger.info('Fetching data from Snowflake')
                    df = pd.DataFrame(conn.cursor().execute(query).fetch_pandas_all())
                    df.reset_index(drop=True, inplace=True)
                    logger.info(f'Successfully fetched {len(df)} rows from Snowflake')
                    return df
                except Exception as e:
                    logger.error(f'Failed to fetch data from Snowflake: {e}')
                    raise e
        except Exception as e:
            logger.error(f'Failed to establish Snowflake connection for data fetching: {e}')
            # Return empty DataFrame as fallback instead of crashing
            logger.warning('Returning empty DataFrame as fallback')
            return pd.DataFrame()

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
