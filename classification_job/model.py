import os
import numpy as np
import pandas as pd
import requests
import textwrap
import re
import time
import boto3
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

from config import CATEGORIES

logging.basicConfig(level=logging.WARNING)

logger = logging.getLogger(__name__)

logging.getLogger("backoff").addHandler(logging.StreamHandler())
logging.getLogger("backoff").setLevel(logging.WARNING)

load_dotenv()

# Track retry count with a mutable object
retry_tracker = {"count": 0}

env = os.getenv('FDF_ENV', 'dev')


# secret_manager_access_key = os.getenv('SECRET_MANAGER_ACCESS_KEY')
# secret_manager_secret_key = os.getenv('SECRET_MANAGER_SECRET_KEY')


# Define a backoff handler to log retry attempts and update count
def log_retry(details):
    retry_tracker["count"] = details["tries"]  # Update retry count
    logger.info(
        "Retrying function %s for the %d time after %0.2f seconds delay due to %s",
        details["target"].__name__,
        details["tries"],
        details["wait"],
        details["exception"],
    )


def get_claude_endpoint_credentials() -> dict:
    client = boto3.client(
        'secretsmanager',
        # aws_access_key_id=secret_manager_access_key,
        # aws_secret_access_key=secret_manager_secret_key,
        region_name='eu-west-1',
    )

    response = client.get_secret_value(SecretId=f'vapafpdf-{env}-deep_claude_endpoint_credentials')

    secret = response['SecretString']
    secret_dict = json.loads(secret)

    service_account_password = secret_dict['client_id']
    service_account_username = secret_dict['client_secret']

    credentials = {
        'client_id': service_account_password,
        'client_secret': service_account_username,
    }

    return credentials


class ConnectionAPI:
    claude_credentials = get_claude_endpoint_credentials()

    client_id = claude_credentials['client_id']
    client_secret = claude_credentials['client_secret']

    @classmethod
    def _get_credits(cls):
        """
        Returns all the credits necessary to access to the Claude endpoint.
        
        Returns:
            token_url -> str: The URL of the token.
            data -> dict: Dict with all the credits.
        
        """
        token_url = 'https://cognito-auth.aiplat.aws.pmicloud.biz/oauth2/token'
        client_id = cls.client_id
        client_secret = cls.client_secret
        scope = 'https://aiplat.aws.pmicloud.biz/tenant https://aiplat.aws.pmicloud.biz/vapafpdf-dev'
        data = {
            'grant_type': 'client_credentials',
            'client_id': client_id,
            'client_secret': client_secret,
            'scope': scope
        }

        return token_url, data

    @classmethod
    def get_access_token(cls):
        """_summary_

        Args:
            token_url (_type_): _description_
            data (_type_): _description_

        Raises:
            ValueError: _description_
            KeyError: _description_

        Returns:
            _type_: _description_
        """
        try:
            token_url, data = ConnectionAPI._get_credits()
            token_response = requests.post(token_url, data=data)
            token_response.raise_for_status()
            access_token = token_response.json().get('access_token')
            if not access_token:
                logger.error("Authentication failed: Access token not retrieved.")
                raise ValueError("Failed to retrieve access token.")
            else:
                logger.info("Access token retrieved successfully.")
                return access_token
        except requests.exceptions.RequestException as e:
            return {"error": "Failed to retrieve access token", "details": str(e)}

    @staticmethod
    def api_chat_completions(prompt, model: str = "anthropic.claude-3-5-sonnet-20240620-v1:0", max_tokens: int = 4096,
                             temperature: float = 0.1, top_p: int = 1):
        """
        Send a prompt to an AI language model API to generate a report based on customer complaint data.

        This function constructs a prompt from a filtered DataFrame of customer complaint data and sends it to an
        AI language model API for generating a comprehensive report. The API request is configured with customizable
        parameters to adjust response length, creativity, and randomness, allowing for fine-tuning of the generated output.

        Parameters:
        -----------
        filtered_df : pandas.DataFrame
            A DataFrame pre-filtered to include only the relevant samples necessary for report generation.
        model : str, optional
            The identifier for the model to be used, defaulting to `"anthropic.claude-3-5-sonnet-20240620-v1:0"`.
        max_tokens : int, optional
            Maximum number of tokens allowed in the generated response. Defaults to 4096.
        temperature : float, optional
            Controls the "creativity" of the output. Higher values increase randomness, with a default of 0.1.
        top_p : float, optional
            Controls the diversity of the output by adjusting the likelihood threshold for token sampling. Defaults to 1.

        Returns:
        --------
        dict
            A dictionary containing the API response data if successful, or an error message and details if an exception occurs.

        Raises:
        -------
        ValueError
            If an access token cannot be retrieved from the token URL.
        requests.exceptions.RequestException
            For network-related errors during the API request.
        """
        try:
            # Retrieve API token
            access_token = ConnectionAPI.get_access_token()

            # Prepare headers and payload for the request
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                "accept": "application/json"
            }
            payload = {
                "inferenceConfig": {
                    "maxTokens": max_tokens,
                    "temperature": temperature,
                    "topP": top_p
                },
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": prompt}]
                    }
                ],
                "modelId": model
            }

            # Send the request to the model API
            try:
                response = requests.post(
                    "https://aiplat.aws.pmicloud.biz/api/bedrock-runtime/v1/converse",
                    headers=headers,
                    json=payload
                )

                response.raise_for_status()
                return response.json()  # Successful response

            except requests.exceptions.RequestException as e:
                return {"error": "Failed to send request to the model API", "details": str(e)}

        except ValueError as ve:
            return {"error": "Value error occurred", "details": str(ve)}
        except Exception as e:
            return {"error": "An unexpected error occurred", "details": str(e)}


class Classification(ConnectionAPI):

    @classmethod
    def create_prompt_classif(cls, sample, categories=CATEGORIES):
        """
        Create a prompt for classifying a customer support ticket.

        Args:
            sample (pd.Series): A pandas Series containing information about the support ticket.
                - 'FORM_TYPE' (str): The type of form (e.g., 'Brand sourcing', 'Consumer feedback').
                - 'FIELD_INTELLIGENCE' (str): The customer support ticket text.
                - Additional fields based on 'FORM_TYPE' (e.g., 'BRAND_NAME_FROM', 'PRODUCT_CATEGORY_NAME').
            categories (str): The available categories for classification.

        Returns:
            str: A formatted prompt for classification.
        """
        form_type = sample['FORM_TYPE']
        ticket = sample['FIELD_INTELLIGENCE']

        # Dictionary to map 'FORM_TYPE' to the required fields
        form_type_mapping = {
            "BRAND_SOURCING": [
                f"Brand Name (From): {sample.get('BRAND_NAME_FROM', '')}",
                f"Brand Name (To): {sample.get('BRAND_NAME_TO', '')}"
            ],
            "CONSUMER_FEEDBACK": [
                f"Product Category Name: {sample.get('PRODUCT_CATEGORY_NAME', '')}",
                f"PMI Product Name: {sample.get('PMI_PRODUCT_NAME', '')}"
            ],
            "CROSS_CATEGORY": [
                f"Product Category Name: {sample.get('PRODUCT_CATEGORY_NAME', '')}"
            ],
            "TOBACCO_CATEGORY": [
                f"Product Category Name: {sample.get('PRODUCT_CATEGORY_NAME', '')}",
                f"TMO Name: {sample.get('TMO_NAME', '')}",
                f"Brand Name: {sample.get('BRAND_NAME', '')}"
            ]
        }

        # Get extra information based on 'FORM_TYPE'
        extra_info = "Additional Context:\n" + "\n".join(
            form_type_mapping.get(form_type, [])) if form_type in form_type_mapping else ""

        # Define the prompt template
        prompt = textwrap.dedent(f"""
            Act as an expert feedback classifier. 
            Your task is to analyze the feedback provided by employees about their interactions with customers and assign the single most appropriate category label from the list below.

            Categories:
            {categories}
            
            Customer Support Ticket:
            {ticket}
            
            {extra_info}
            
            Instructions:
            Respond only with the exact label of the best-fitting category.
            Return your answer in this precise format: <category>{{{{category_label}}}}</category>
            Do not include any explanations, comments, or additional text.
        """)
        return prompt

    @classmethod
    def create_prompt_translated(cls, sample, categories=CATEGORIES):
        """
        Create a prompt for classifying a customer support ticket.

        Args:
            sample (pd.Series): A pandas Series containing information about the support ticket.
                - 'FORM_TYPE' (str): The type of form (e.g., 'Brand sourcing', 'Consumer feedback').
                - 'FIELD_INTELLIGENCE' (str): The customer support ticket text.
                - Additional fields based on 'FORM_TYPE' (e.g., 'BRAND_NAME_FROM', 'PRODUCT_CATEGORY_NAME').
            categories (str): The available categories for classification.

        Returns:
            str: A formatted prompt for classification.
        """
        form_type = sample['FORM_TYPE']
        ticket = sample['FIELD_INTELLIGENCE_TRANSLATED']

        # Dictionary to map 'FORM_TYPE' to the required fields
        form_type_mapping = {
            "BRAND_SOURCING": [
                f"Brand Name (From): {sample.get('BRAND_NAME_FROM', '')}",
                f"Brand Name (To): {sample.get('BRAND_NAME_TO', '')}"
            ],
            "CONSUMER_FEEDBACK": [
                f"Product Category Name: {sample.get('PRODUCT_CATEGORY_NAME', '')}",
                f"PMI Product Name: {sample.get('PMI_PRODUCT_NAME', '')}"
            ],
            "CROSS_CATEGORY": [
                f"Product Category Name: {sample.get('PRODUCT_CATEGORY_NAME', '')}"
            ],
            "TOBACCO_CATEGORY": [
                f"Product Category Name: {sample.get('PRODUCT_CATEGORY_NAME', '')}",
                f"TMO Name: {sample.get('TMO_NAME', '')}",
                f"Brand Name: {sample.get('BRAND_NAME', '')}"
            ]
        }

        # Get extra information based on 'FORM_TYPE'
        extra_info = "Additional Context:\n" + "\n".join(
            form_type_mapping.get(form_type, [])) if form_type in form_type_mapping else ""

        # Define the prompt template
        prompt = textwrap.dedent(f"""
            Act as an expert feedback classifier. 
            Your task is to analyze the feedback provided by employees about their interactions with customers and assign the single most appropriate category label from the list below.

            Categories:
            {categories}
            
            Customer Support Ticket:
            {ticket}
            
            {extra_info}
            
            Instructions:
            Respond only with the exact label of the best-fitting category.
            Return your answer in this precise format: <category>{{{{category_label}}}}</category>
            Do not include any explanations, comments, or additional text.
        """)
        return prompt

    @classmethod
    def extract_category_value(cls, text):
        """
        Extracts the value between <category> tags in the given text.
        
        Parameters:
            text (str): The text containing the <category> tag.
        
        Returns:
            str: The value inside the <category> tag.
        """
        match = re.search(r"<category>(.*?)</category>", text)
        if match:
            return match.group(1)
        else:
            return None

    @classmethod
    def get_class(cls, row, retries=3, delay=2):
        """
        Calls the `api_chat_completions` function and extracts the classification result.
        
        Parameters:
            row (pd.Series): A single row from the DataFrame.
        
        Returns:
            str: The classification result extracted from the API response, or `None` if an error occurs.
        """
        if not pd.isna(row.get('CLASS')):  # Skip rows where 'CLASS' is already filled
            return row['CLASS']

        attempt = 0
        while attempt < retries:
            try:
                # Create the prompt
                prompt = Classification.create_prompt_classif(row)

                # Call the API function
                response = ConnectionAPI.api_chat_completions(prompt)

                # If 'output' is missing and there's a network error message, treat it as retryable
                if 'message' in response and 'Network error' in response['message']:
                    print(f"Network error in response for row {row.name}, retrying ({attempt + 1})...")
                    attempt += 1
                    time.sleep(delay)
                    continue

                classification = response['output']['message']['content'][0]['text']

                return Classification.extract_category_value(classification)

            except requests.RequestException as e:
                print(f"Network error on attempt {attempt + 1} for row {row.name}: {e}")
                attempt += 1
                time.sleep(delay)

            except (KeyError, IndexError) as e:
                # Print the full row for debugging
                print(f"Error for row {row.name}: {e}, retrying ({attempt + 1})...")
                attempt += 1
                time.sleep(delay)

        # After all retries fail, attempt one last time with `create_prompt_translated`
        print(f"Final attempt for row {row.name} using translated prompt.")
        try:
            # Create the prompt using `create_prompt_translated`
            prompt_translated = Classification.create_prompt_translated(row)

            # Call the API function
            response = ConnectionAPI.api_chat_completions(prompt_translated)

            # If successful, return the classification
            classification = response['output']['message']['content'][0]['text']
            return Classification.extract_category_value(classification)

        except requests.RequestException as e:
            print(f"Final attempt network error for row {row.name}: {e}")

        except (KeyError, IndexError) as e:
            print(f"Final attempt error for row {row.name}: {e}")

        # If all attempts fail, log the failure
        print(f"Failed to process row {row.name} after all attempts.")
        return None

    @classmethod
    def classify_chunk(cls, df_chunk):
        for idx, row in df_chunk.iterrows():
            if pd.isna(row['CLASS']):
                df_chunk.at[idx, 'CLASS'] = cls.get_class(row)
        return df_chunk

    @classmethod
    def process_dataframe(cls, df, num_workers=4):
        logger.info("Process to the classification of the Dataframe...")

        df_chunks = np.array_split(df, num_workers)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(cls.classify_chunk, df_chunks))

        df_final = pd.concat(results, ignore_index=True)

        logger.info("Classification completed.")

        return df_final
