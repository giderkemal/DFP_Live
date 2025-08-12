import pandas as pd
import os
import requests
import re
import json
import time
import numpy as np
from datetime import datetime
import tiktoken
import streamlit as st
import logging
import backoff
import xml.etree.ElementTree as ET
from typing import List, Dict, Generator
from dotenv import load_dotenv

from base_matrix import BaseMatrixService
from chat import stream_message_from_anthropic
from streamlit_utils.session import StreamlitStateManager, Message
from config import *

logging.basicConfig(level=logging.WARNING)

# Initialize logger
logger = logging.getLogger(__name__)

# For backoff logging
logging.getLogger("backoff").addHandler(logging.StreamHandler())
logging.getLogger("backoff").setLevel(logging.WARNING)

# Track retry count with a mutable object
retry_tracker = {"count": 0}

load_dotenv()


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


class ConnectionAPI:
    CLIENT_ID = os.getenv("CLIENT_ID")
    CLIENT_SECRET = os.getenv("CLIENT_SECRET")
    TOKEN_URL = os.getenv("BEDROCK_TOKEN_URL")

    @classmethod
    def _get_credits(cls):
        """
        Returns all the credits necessary to access to the Claude endpoint.

        Returns:
            token_url -> str: The URL of the token.
            data -> dict: Dict with all the credits.

        """
        token_url = cls.TOKEN_URL
        client_id = cls.CLIENT_ID
        client_secret = cls.CLIENT_SECRET
        scope = "https://aiplat.aws.pmicloud.biz/tenant https://aiplat.aws.pmicloud.biz/vapafpdf-dev"
        data = {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": scope,
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
            access_token = token_response.json().get("access_token")
            if not access_token:
                # logger.error("Authentication failed: Access token not retrieved.")
                raise ValueError("Failed to retrieve access token.")
            else:
                # logger.info("Access token retrieved successfully.")
                return access_token
        except requests.exceptions.RequestException as e:
            return {"error": "Failed to retrieve access token", "details": str(e)}

    @staticmethod
    def api_chat_completions_stream(
        prompt,
        model: str = "anthropic.claude-3-5-sonnet-20240620-v1:0",
        max_tokens: int = 4096,
        temperature: float = 0.1,
        top_p: int = 1,
    ):
        try:
            # Retrieve API token
            access_token = ConnectionAPI.get_access_token()

            # Prepare headers and payload for the request
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                "accept": "application/json",
            }
            payload = {
                "inferenceConfig": {
                    "maxTokens": max_tokens,
                    "temperature": temperature,
                    "topP": top_p,
                },
                "messages": [{"role": "user", "content": [{"text": prompt}]}],
                "modelId": model,
            }

            # Set Private to False
            private_api = False

            if private_api:
                url = "https://aiplat.aws.private-pmideep.biz/api/bedrock-runtime/v1/converse_stream"
            else:
                url = "https://aiplat.aws.pmicloud.biz/api/bedrock-runtime/v1/converse_stream"

            # Send the request to the model API
            try:
                start_response = requests.post(
                    f"{url}/start", headers=headers, json=payload
                )
                # with st.expander("Response:", expanded=False):
                #    st.info(start_response.json())
                start_response.raise_for_status()

                try:
                    stream_id = (
                        start_response.json()
                        .get("ResponseMetadata", {})
                        .get("RequestId")
                    )
                except (AttributeError, ValueError, json.JSONDecodeError) as e:
                    logger.error(
                        "Stream initiation failed: Stream ID not retrieved. Details:", e
                    )
                    raise ValueError(
                        "Stream initiation failed: Stream ID not retrieved. Details:", e
                    )

                try:
                    routing_cookie = start_response.cookies.get("routing")
                    cookies = {"routing": routing_cookie}
                except (AttributeError, ValueError, json.JSONDecodeError) as e:
                    logger.error(
                        "Stream initiation failed: Routing cookie not retrieved. Details:",
                        e,
                    )
                    raise ValueError(
                        "Stream initiation failed: Routing cookie not retrieved. Details: ",
                        e,
                    )

                def backoff_hdlr(details):
                    print(
                        "Backing off {wait:0.1f} seconds after {tries} tries "
                        "calling function {target} with args {args} and kwargs "
                        "{kwargs}".format(**details)
                    )

                def fatal_code(e):
                    return 400 <= e.response.status_code < 500

                # Generator function for streaming data with exponential backoff retry
                @backoff.on_exception(
                    backoff.expo,
                    requests.exceptions.RequestException,
                    logger=logger,
                    max_time=60,
                    max_tries=5,
                    giveup=fatal_code,
                    raise_on_giveup=True,
                    on_backoff=backoff_hdlr,
                )
                def response_generator() -> Generator[Dict, None, None]:
                    next_url = f"{url}/next/{stream_id}"
                    logger.info(
                        "Backoff: Current retry attempt: %d", retry_tracker["count"]
                    )
                    while True:
                        # Get next chunk of data
                        start_time = time.time()
                        next_response = requests.get(
                            next_url, headers=headers, cookies=cookies
                        )
                        elapsed_time = time.time() - start_time
                        server_time = (
                            next_response.json()
                            .get("api_metrics")
                            .get("total_response_time")
                        )
                        logger.debug(
                            "Request to %s took:\n Total: %0.2f seconds: Server response time: {server_time:3f}".format(
                                server_time=server_time
                            ),
                            next_url,
                            elapsed_time,
                        )
                        if next_response.json().get("event_type") == "message_stop":
                            logger.info("Stream has ended.")
                            break
                        else:
                            # Yield the JSON response
                            yield next_response.json()

                try:
                    return response_generator
                except requests.RequestException as e:
                    logger.error("An error occurred: %s", e)
                    # logger.exception("Exception occurred")
                    # raise e

            except requests.exceptions.RequestException as e:
                return {
                    "error": "Failed to send request to the model API",
                    "details": str(e),
                }

        except ValueError as ve:
            return {"error": "Value error occurred", "details": str(ve)}
        except Exception as e:
            return {"error": "An unexpected error occurred", "details": str(e)}

    @staticmethod
    def api_chat_completions_stream_message(
        messages,
        system_prompt,
        model: str = "anthropic.claude-3-5-sonnet-20240620-v1:0",
        max_tokens: int = 4096,
        temperature: float = 0.1,
        top_p: int = 1,
    ):
        try:
            # Retrieve API token
            access_token = ConnectionAPI.get_access_token()

            # Prepare headers and payload for the request
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                "accept": "application/json",
            }
            payload = {
                "inferenceConfig": {
                    "maxTokens": max_tokens,
                    "temperature": temperature,
                    "topP": top_p,
                },
                "system": [{"text": system_prompt}],
                "messages": messages,
                "modelId": model,
            }

            # Set Private to False
            private_api = False

            if private_api:
                url = "https://aiplat.aws.private-pmideep.biz/api/bedrock-runtime/v1/converse_stream"
            else:
                url = "https://aiplat.aws.pmicloud.biz/api/bedrock-runtime/v1/converse_stream"

            # Send the request to the model API
            try:
                start_response = requests.post(
                    f"{url}/start", headers=headers, json=payload
                )
                print(start_response.json())
                start_response.raise_for_status()

                try:
                    stream_id = (
                        start_response.json()
                        .get("ResponseMetadata", {})
                        .get("RequestId")
                    )
                except (AttributeError, ValueError, json.JSONDecodeError) as e:
                    logger.error(
                        "Stream initiation failed: Stream ID not retrieved. Details:", e
                    )
                    raise ValueError(
                        "Stream initiation failed: Stream ID not retrieved. Details:", e
                    )

                try:
                    routing_cookie = start_response.cookies.get("routing")
                    cookies = {"routing": routing_cookie}
                except (AttributeError, ValueError, json.JSONDecodeError) as e:
                    logger.error(
                        "Stream initiation failed: Routing cookie not retrieved. Details:",
                        e,
                    )
                    raise ValueError(
                        "Stream initiation failed: Routing cookie not retrieved. Details: ",
                        e,
                    )

                def backoff_hdlr(details):
                    print(
                        "Backing off {wait:0.1f} seconds after {tries} tries "
                        "calling function {target} with args {args} and kwargs "
                        "{kwargs}".format(**details)
                    )

                def fatal_code(e):
                    return 400 <= e.response.status_code < 500

                # Generator function for streaming data with exponential backoff retry
                @backoff.on_exception(
                    backoff.expo,
                    requests.exceptions.RequestException,
                    logger=logger,
                    max_time=60,
                    max_tries=5,
                    giveup=fatal_code,
                    raise_on_giveup=True,
                    on_backoff=backoff_hdlr,
                )
                def response_generator() -> Generator[Dict, None, None]:
                    next_url = f"{url}/next/{stream_id}"
                    logger.info(
                        "Backoff: Current retry attempt: %d", retry_tracker["count"]
                    )
                    while True:
                        # Get next chunk of data
                        start_time = time.time()
                        next_response = requests.get(
                            next_url, headers=headers, cookies=cookies
                        )
                        elapsed_time = time.time() - start_time
                        server_time = (
                            next_response.json()
                            .get("api_metrics")
                            .get("total_response_time")
                        )
                        logger.debug(
                            "Request to %s took:\n Total: %0.2f seconds: Server response time: {server_time:3f}".format(
                                server_time=server_time
                            ),
                            next_url,
                            elapsed_time,
                        )
                        if next_response.json().get("event_type") == "message_stop":
                            logger.info("Stream has ended.")
                            break
                        else:
                            # Yield the JSON response
                            yield next_response.json()

                try:
                    return response_generator
                except requests.RequestException as e:
                    logger.error("An error occurred: %s", e)
                    # logger.exception("Exception occurred")
                    # raise e

            except requests.exceptions.RequestException as e:
                return {
                    "error": "Failed to send request to the model API",
                    "details": str(e),
                }

        except ValueError as ve:
            return {"error": "Value error occurred", "details": str(ve)}
        except Exception as e:
            return {"error": "An unexpected error occurred", "details": str(e)}

    @staticmethod
    def api_chat_completions(
        prompt,
        model: str = "anthropic.claude-3-5-sonnet-20240620-v1:0",
        max_tokens: int = 4096,
        temperature: float = 0.1,
        top_p: int = 1,
    ):
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
                "accept": "application/json",
            }
            payload = {
                "inferenceConfig": {
                    "maxTokens": max_tokens,
                    "temperature": temperature,
                    "topP": top_p,
                },
                "messages": [{"role": "user", "content": [{"text": prompt}]}],
                "modelId": model,
            }

            # Send the request to the model API
            try:
                response = requests.post(
                    "https://aiplat.aws.pmicloud.biz/api/bedrock-runtime/v1/converse",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                return response.json()  # Successful response

            except requests.exceptions.RequestException as e:
                return {
                    "error": "Failed to send request to the model API",
                    "details": str(e),
                }

        except ValueError as ve:
            return {"error": "Value error occurred", "details": str(ve)}
        except Exception as e:
            return {"error": "An unexpected error occurred", "details": str(e)}

    @staticmethod
    def aggregate_response(generator):
        return "".join(
            message["delta"]["text"]
            for message in generator()
            if "delta" in message and "text" in message["delta"]
        )

    @staticmethod
    def get_token_counts(response):
        """
        Extracts the input and output token counts from a JSON response.

        Parameters:
        response (requests.Response): The response object obtained from a request.

        Returns:
        tuple: A tuple containing the input token count and output token count.
        """
        try:
            # Extract the token counts from the response
            input_tokens = response["usage"]["inputTokens"]
            output_tokens = response["usage"]["outputTokens"]

            return input_tokens, output_tokens
        except (AttributeError, KeyError, json.JSONDecodeError) as e:
            # Return None if the expected keys are not found
            print("Error extracting token counts:", e)
            return None, None

    def count_tokens_with_anthropic(
        messages, system_prompt, model_name="gpt-3.5-turbo"
    ):
        # Get the tokenizer for the model
        encoding = tiktoken.encoding_for_model(model_name)

        # Count tokens for the system prompt
        total_tokens = len(encoding.encode(system_prompt))
        for message in messages:
            # Count tokens for the role
            role_tokens = len(encoding.encode(message.role))

            # Check if content is a string, and handle it accordingly
            if isinstance(message.content, str):
                content_tokens = len(encoding.encode(message.content))
            else:
                # Handle other possible structures of `content` if needed
                content_tokens = 0

            # Add the tokens for this message
            total_tokens += role_tokens + content_tokens

        return total_tokens

    @staticmethod
    def calculate_price(
        response,
        input_price_per_million_tokens=3.0,
        output_price_per_million_tokens=15.0,
    ):
        """
        Calculates the total price based on token usage from a Claude Sonnet API response.

        Parameters:
        response (requests.Response): The response object obtained from a request.
        input_price_per_million_tokens (float): Cost per million input tokens. Default is 3.0.
        output_price_per_million_tokens (float): Cost per million output tokens. Default is 15.0.

        Returns:
        float: The calculated total price, or None if there is an error in processing the response.
        """
        try:
            # Extract token counts
            input_tokens, output_tokens = ConnectionAPI.get_token_counts(response)

            # Validate tokens exist and are numeric
            if input_tokens is None or output_tokens is None:
                raise KeyError("Missing 'inputTokens' or 'outputTokens' in response.")

            # Calculate costs
            input_price = (input_tokens / 1_000_000) * input_price_per_million_tokens
            output_price = (output_tokens / 1_000_000) * output_price_per_million_tokens
            total_price = input_price + output_price

            return total_price

        except (KeyError, TypeError, json.JSONDecodeError) as e:
            print(f"Error calculating price: {e}")
            return None


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
        form_type = sample["FORM_TYPE"]
        ticket = sample["FIELD_INTELLIGENCE"]

        # Dictionary to map 'FORM_TYPE' to the required fields
        form_type_mapping = {
            "BRAND_SOURCING": [
                f"Brand Name (From): {sample.get('BRAND_NAME_FROM', '')}",
                f"Brand Name (To): {sample.get('BRAND_NAME_TO', '')}",
            ],
            "CONSUMER_FEEDBACK": [
                f"Product Category Name: {sample.get('PRODUCT_CATEGORY_NAME', '')}",
                f"PMI Product Name: {sample.get('PMI_PRODUCT_NAME', '')}",
            ],
            "CROSS_CATEGORY": [
                f"Product Category Name: {sample.get('PRODUCT_CATEGORY_NAME', '')}"
            ],
            "TOBACCO_CATEGORY": [
                f"Product Category Name: {sample.get('PRODUCT_CATEGORY_NAME', '')}",
                f"TMO Name: {sample.get('TMO_NAME', '')}",
                f"Brand Name: {sample.get('BRAND_NAME', '')}",
            ],
        }

        # Get extra information based on 'FORM_TYPE'
        extra_info = (
            "Additional Context:\n" + "\n".join(form_type_mapping.get(form_type, []))
            if form_type in form_type_mapping
            else ""
        )

        # Define the prompt template
        prompt = textwrap.dedent(
            f"""
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
        """
        )
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
        form_type = sample["FORM_TYPE"]
        ticket = sample["FIELD_INTELLIGENCE_TRANSLATED"]

        # Dictionary to map 'FORM_TYPE' to the required fields
        form_type_mapping = {
            "BRAND_SOURCING": [
                f"Brand Name (From): {sample.get('BRAND_NAME_FROM', '')}",
                f"Brand Name (To): {sample.get('BRAND_NAME_TO', '')}",
            ],
            "CONSUMER_FEEDBACK": [
                f"Product Category Name: {sample.get('PRODUCT_CATEGORY_NAME', '')}",
                f"PMI Product Name: {sample.get('PMI_PRODUCT_NAME', '')}",
            ],
            "CROSS_CATEGORY": [
                f"Product Category Name: {sample.get('PRODUCT_CATEGORY_NAME', '')}"
            ],
            "TOBACCO_CATEGORY": [
                f"Product Category Name: {sample.get('PRODUCT_CATEGORY_NAME', '')}",
                f"TMO Name: {sample.get('TMO_NAME', '')}",
                f"Brand Name: {sample.get('BRAND_NAME', '')}",
            ],
        }

        # Get extra information based on 'FORM_TYPE'
        extra_info = (
            "Additional Context:\n" + "\n".join(form_type_mapping.get(form_type, []))
            if form_type in form_type_mapping
            else ""
        )

        # Define the prompt template
        prompt = textwrap.dedent(
            f"""
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
        """
        )
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
        Calls the `api_chat_completions_sample` function and extracts the classification result.

        Parameters:
            row (pd.Series): A single row from the DataFrame.

        Returns:
            str: The classification result extracted from the API response, or `None` if an error occurs.
        """
        if not pd.isna(row.get("CLASS")):  # Skip rows where 'CLASS' is already filled
            return row["CLASS"]

        attempt = 0
        while attempt < retries:
            try:
                # Create the prompt
                prompt = Classification.create_prompt_classif(row)

                # Call the API function
                response = ConnectionAPI.api_chat_completions(prompt)

                # If 'output' is missing and there's a network error message, treat it as retryable
                if "message" in response and "Network error" in response["message"]:
                    print(
                        f"Network error in response for row {row.name}, retrying ({attempt + 1})..."
                    )
                    attempt += 1
                    time.sleep(delay)
                    continue

                classification = response["output"]["message"]["content"][0]["text"]
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
            classification = response["output"]["message"]["content"][0]["text"]
            return Classification.extract_category_value(classification)

        except requests.RequestException as e:
            print(f"Final attempt network error for row {row.name}: {e}")

        except (KeyError, IndexError) as e:
            print(f"Final attempt error for row {row.name}: {e}")

        # If all attempts fail, log the failure
        print(f"Failed to process row {row.name} after all attempts.")
        return None

    @classmethod
    def process_dataframe_and_save(
        cls, df, chunk_size=50, output_file="../data/df_final_classified.csv"
    ):

        # Ensure the 'CLASS' column exists
        if "CLASS" not in df.columns:
            print("Adding missing 'CLASS' column to the DataFrame...")
            df["CLASS"] = None  # Initialize the column with None values

        for idx, row in df.iterrows():
            # Process only rows where 'CLASS' is NaN/None
            if pd.isna(row["CLASS"]):
                df.loc[idx, "CLASS"] = Classification.get_class(row)

            # Save progress every `chunk_size` rows
            if idx % chunk_size == 0:
                print(f"Saving progress at row {idx}...")
                df.to_csv(output_file, index=False)

        # Final save
        print("Saving final DataFrame...")
        df.to_csv(output_file, index=False)


class ReportGeneration(ConnectionAPI):

    @classmethod
    def create_prompt_report_location_filtered(cls, df):
        """
        Generate a structured prompt for a language model to create a detailed report based on customer complaint data.

        This function constructs a prompt intended for a language model, such as GPT, to generate a comprehensive
        report analyzing customer complaints. The report focuses on extracting insights into common complaint categories,
        identifying patterns, and suggesting actionable recommendations. The prompt includes specific instructions for
        structuring the report, with details on sections like an executive summary, category breakdown, and recommendations.

        Parameters:
        -----------
        df : pandas.DataFrame
            A filtered DataFrame containing relevant customer complaint data. It is expected to include
            fields required for generating the report (e.g., translated field intelligence, complaint class, location,
            and terminal information).

        Returns:
        --------
        str
            A formatted prompt containing structured instructions for the language model, which includes:
            - **Executive Summary**: Brief overview of main findings (top complaint categories, trends, and urgent issues).
            - **Category Breakdown**: Segmented analysis by predefined categories, summarizing key trends, recurring issues, and complaint volume.
            - **Specific Insights & Patterns**: Highlights frequent customer pain points, emerging themes, and possible external influences.
            - **Suggestions & Actionable Recommendations**: Practical recommendations to improve customer satisfaction, tailored to each complaint category, prioritized by urgency.
            - **Visuals**: Request for relevant visuals, such as graphs, charts, or tables, to enhance understanding of complaint distribution and trends.

        Notes:
        ------
        - Utilizes the `ReportGeneration.create_list_infos` function to preprocess `df` and structure the data as a JSON-like
        format that is included in the prompt as input for the language model.
        - The prompt emphasizes a reader-friendly and professional format, tailored for a business audience with a report length limit of approximately 1,500 words.
        """

        data = ReportGeneration.create_xml_infos(df)

        based_prompt = f"""Act as a data analyst specializing in employee feedback and customer request management. Your objective is to create a comprehensive and actionable report that evaluates the current state of global duty-free operations, leveraging structured data from employee feedback and customer-related comments. The insights you provide will directly guide teams in addressing operational challenges through targeted actions.

            ### <task overview>
            You are tasked to analyze the provided data and generate a detailed report that addresses the following key aspects:

            1. **Top 5 Global Operational Challenges**
            - Identify and list the top 5 issues affecting global duty-free operations.
            - Categorize these issues into specific areas (e.g., inventory management, service quality, customer behavior).
            - Support findings with precise examples from the data, citing relevant Field Intelligence and Row_ID values.
            - Propose concrete, actionable recommendations to resolve or mitigate each challenge (e.g., policy adjustments, additional training, resource reallocation).

            2. **Monthly Trends and Changes Over Time**:
            - Analyze the data to identify monthly trends over the year.
            - Highlight any significant changes in feedback or operational issues from month to month.
            - Include examples to illustrate trends (e.g., "Inventory complaints spiked by 30% in December due to holiday demand").
            - Suggest time-sensitive actions to address recurring or seasonal patterns (e.g., preparing for known spikes in demand).

            ### **<instructions>**
            The report must include the following elements:
            1. **Top 5 Problems or Insights**:
            - Present these as a **numbered list** with concise, impactful descriptions.
            - Highlight key metrics (e.g., "40% of feedback in February focused on delays in order fulfillment").
            - Include actionable solutions for each insight.
            - Cite supporting examples using Field Intelligence and Row_ID.

            2. **Monthly Trends Analysis**:
            - Summarize changes in feedback trends over the year.
            - Highlight significant monthly variations or emerging patterns.
            - Provide actionable recommendations to address identified trends.
            - Stick to plain text rather than bullet points

            3. <output format>:
            The output should be in XML format:
                    <response>
                        <report>
                            Complete report as a single text block with full structure and formatting
                        </report>
                        <examplesID>
                            A list of the Row_IDs corresponding to the specific examples cited in the report, in the order they appear.
                        </examplesID>
                    </response>
            </output format>
            </instructions>


            5. <formatting instructions>:
            - Provide specific, data-driven insights supported by exact examples.
            - Emphasize key metrics and trends in a clear and impactful way.
            - Ensure all recommendations are actionable and directly tied to the identified challenges.
            </formatting instructions>

            Step-by-Step Process:
            1. Analyze the dataset to identify the most pressing recurring challenges.
            2. Examine monthly trends in the data to uncover significant changes or emerging patterns.
            3. Quantify the scale and impact of problems using metrics and examples.
            4. For each insight, provide supporting examples, key metrics, and actionable solutions.

            Use this data to generate the report:
            <data>
            {data}
            </data>

            Lets think step by step.
            """

        return based_prompt

    @classmethod
    def create_prompt_report_time_filtered(cls, df):
        """
        Generate a structured prompt for a language model to create a detailed report based on customer complaint data.

        This function constructs a prompt intended for a language model, such as GPT, to generate a comprehensive
        report analyzing customer complaints. The report focuses on extracting insights into common complaint categories,
        identifying patterns, and suggesting actionable recommendations. The prompt includes specific instructions for
        structuring the report, with details on sections like an executive summary, category breakdown, and recommendations.

        Parameters:
        -----------
        df : pandas.DataFrame
            A filtered DataFrame containing relevant customer complaint data. It is expected to include
            fields required for generating the report (e.g., translated field intelligence, complaint class, location,
            and terminal information).

        Returns:
        --------
        str
            A formatted prompt containing structured instructions for the language model, which includes:
            - **Executive Summary**: Brief overview of main findings (top complaint categories, trends, and urgent issues).
            - **Category Breakdown**: Segmented analysis by predefined categories, summarizing key trends, recurring issues, and complaint volume.
            - **Specific Insights & Patterns**: Highlights frequent customer pain points, emerging themes, and possible external influences.
            - **Suggestions & Actionable Recommendations**: Practical recommendations to improve customer satisfaction, tailored to each complaint category, prioritized by urgency.
            - **Visuals**: Request for relevant visuals, such as graphs, charts, or tables, to enhance understanding of complaint distribution and trends.

        Notes:
        ------
        - Utilizes the `ReportGeneration.create_xml_infos` function to preprocess `df` and structure the data as a JSON-like
        format that is included in the prompt as input for the language model.
        - The prompt emphasizes a reader-friendly and professional format, tailored for a business audience with a report length limit of approximately 1,500 words.
        """

        data = ReportGeneration.create_xml_infos(df)

        based_prompt = f"""Act as a data analyst specializing in employee feedback and customer request management. Your objective is to create a comprehensive and actionable report that evaluates the current state of global duty-free operations, leveraging structured data from employee feedback and customer-related comments. The insights you provide will directly guide teams in addressing operational challenges through targeted actions.

            ### <task overview>
            You are tasked to analyze the provided data and generate a detailed report that addresses the following key aspects:

            1. **Top 5 Global Operational Challenges**
            - Identify and list the top 5 issues affecting global duty-free operations.
            - Categorize these issues into specific areas (e.g., inventory management, service quality, customer behavior).
            - Support findings with precise examples from the data, citing relevant Field Intelligence and Row_ID values.

            2. **Monthly Trends and Changes Over Time**:
            - Analyze the data to identify monthly trends over the year.
            - Highlight any significant changes in feedback or operational issues from month to month.
            - Include examples to illustrate trends (e.g., "Inventory complaints spiked by 30% in December due to holiday demand").
            - Suggest time-sensitive actions to address recurring or seasonal patterns (e.g., preparing for known spikes in demand).

            ### **<instructions>**
            The report must include the following elements:
            1. **Top 5 Problems or Insights**:
            - Present these as a **numbered list** with concise, impactful descriptions.
            - Highlight key metrics (e.g., "40% of feedback in February focused on delays in order fulfillment").
            - Include actionable solutions for each insight.
            - Cite supporting examples using Field Intelligence and Row_ID.

            2. **Monthly Trends Analysis**:
            - Summarize changes in feedback trends over the year.
            - Highlight significant monthly variations or emerging patterns.
            - Provide actionable recommendations to address identified trends.
            - Stick to plain text rather than bullet points

            3. <output format>:
            The output should be in XML format:
                    <response>
                        <report>
                            Complete report as a single text block with full structure and formatting
                        </report>
                        <examplesID>
                            A list of the Row_IDs corresponding to the specific examples cited in the report, in the order they appear.
                        </examplesID>
                    </response>
            </output format>
            </instructions>


            5. <formatting instructions>:
            - Provide specific, data-driven insights supported by exact examples.
            - Emphasize key metrics and trends in a clear and impactful way.
            - Ensure all recommendations are actionable and directly tied to the identified challenges.
            </formatting instructions>

            Step-by-Step Process:
            1. Analyze the dataset to identify the most pressing recurring challenges.
            2. Examine monthly trends in the data to uncover significant changes or emerging patterns.
            3. Quantify the scale and impact of problems using metrics and examples.
            4. For each insight, provide supporting examples, key metrics, and actionable solutions.

            Use this data to generate the report:
            <data>
            {data}
            </data>

            Lets think step by step.
            """

        return based_prompt

    @classmethod
    def create_prompt_report(cls, df):
        """
        Generate a structured prompt for a language model to create a detailed report based on customer complaint data.

        This function constructs a prompt intended for a language model, such as GPT, to generate a comprehensive
        report analyzing customer complaints. The report focuses on extracting insights into common complaint categories,
        identifying patterns, and suggesting actionable recommendations. The prompt includes specific instructions for
        structuring the report, with details on sections like an executive summary, category breakdown, and recommendations.

        Parameters:
        -----------
        df : pandas.DataFrame
            A filtered DataFrame containing relevant customer complaint data. It is expected to include
            fields required for generating the report (e.g., translated field intelligence, complaint class, location,
            and terminal information).

        Returns:
        --------
        str
            A formatted prompt containing structured instructions for the language model, which includes:
            - **Executive Summary**: Brief overview of main findings (top complaint categories, trends, and urgent issues).
            - **Category Breakdown**: Segmented analysis by predefined categories, summarizing key trends, recurring issues, and complaint volume.
            - **Specific Insights & Patterns**: Highlights frequent customer pain points, emerging themes, and possible external influences.
            - **Suggestions & Actionable Recommendations**: Practical recommendations to improve customer satisfaction, tailored to each complaint category, prioritized by urgency.
            - **Visuals**: Request for relevant visuals, such as graphs, charts, or tables, to enhance understanding of complaint distribution and trends.

        Notes:
        ------
        - Utilizes the `ReportGeneration.create_xml_infos` function to preprocess `df` and structure the data as a JSON-like
        format that is included in the prompt as input for the language model.
        - The prompt emphasizes a reader-friendly and professional format, tailored for a business audience with a report length limit of approximately 1,500 words.
        """

        data = ReportGeneration.create_xml_infos(df)

        based_prompt = f"""Act as a data analyst specializing in employee feedback and customer request management. Your objective is to create a comprehensive and actionable report that evaluates the current state of global duty-free operations, leveraging structured data from employee feedback and customer-related comments. The insights you provide will directly guide teams in addressing operational challenges through targeted actions.

            ### <task overview>
            You are tasked to analyze the provided data and generate a detailed report that addresses the following key aspects:

            1. **Top 5 Global Operational Challenges**
            - Identify and list the top issues affecting global duty-free operations. Describe them in two or three bullet points.
            - Categorize these issues into specific areas (e.g., inventory management, service quality, customer behavior). Use the 'CLASS' value in the <data>.
            - Support findings with several precise examples from the data, citing relevant Field Intelligence and Row_ID values (at least 3 per Challenge).
            - Quantify the magnitude of each problem (e.g., percentage of negative feedback, frequency of occurrence, or estimated financial impact).
            - Propose concrete, actionable recommendations to resolve or mitigate each challenge (e.g., policy adjustments, additional training, resource reallocation).


            2. **Location-Specific Challenges**
            - Break down the findings by **region or specific duty-free stations**, Describe them in two or three bullet points.
            - Highlight patterns or recurrent issues (e.g., pricing complaints in Region A, staff shortages in Station B).
            - Support observations with **specific data examples** (Field Intelligence and Row_ID) and recommend **location-specific actions** to address these challenges.

            3. **Timing and Seasonal Trends**
            - Analyze feedback for patterns tied to **specific times of the year** (e.g., holidays, weekends, or seasonal events), Describe them in two or three bullet points.
            - Identify spikes in issues or unique challenges during these periods.
            - Provide examples illustrating these trends and suggest proactive measures to prevent recurring problems.
            </task overview>

            ### **<instructions>**
            The report must include the following elements:
            1. **Top 5 Problems or Insights**:
            - Present these as a **numbered list** with concise, impactful descriptions, Describe them in two or three bullet points.
            {GLOBAL_CHALLENGES_FORMAT}

            2. **Location Breakdown**:
            - Summarize feedback trends and challenges for each region or duty-free station, Describe them in two or three bullet points.
            - Propose **location-specific recommendations** tailored to local issues.
            - Cite several (at least 3) examples to support the findings and insights
            {CITATION_FORMAT_PROMPT}

            3. **Timing/Seasonal Trends**:
            - Detail patterns in feedback volume and sentiment over time, Describe them in two or three bullet points.
            - Offer recommendations for addressing time-sensitive challenges effectively.
            - Cite several (at least 3) examples to support the findings and insights
            {CITATION_FORMAT_PROMPT}

            4. <output format>:
            The output should be in XML format:
                    <response>
                        <report>
                            Complete report as a single text block with full structure and formatting
                        </report>
                        <examplesID>
                            A list of the Row_IDs corresponding to the specific examples cited in the report, in the order they appear.
                        </examplesID>
                    </response>
            </output format>
            </instructions>


            5. <formatting instructions>:
            - Provide specific, data-driven insights supported by exact examples. Don't limit yourself to one insight, give several of them.
            - Emphasize key metrics and trends in a clear and impactful way.
            - Ensure all recommendations are actionable and directly tied to the identified challenges.
            </formatting instructions>

            Step-by-Step Process:
            1. Analyze the dataset to identify top problems globally.
            2. Segment the data by location to understand regional differences.
            3. Identify timing patterns, if any, to provide time-sensitive insights.
            4. For each insight or problem, provide examples with supporting metrics and recommendations.

            Use this data to generate the report:
            <data>
            {data}
            </data>

            Lets think step by step.
            """

        return based_prompt

    @classmethod
    def create_generic_prompt_report(cls, df, EXTRA_DEMAND="", EXTRA_INSTRUCTIONS=""):
        """Create Prompt."""

        data = ReportGeneration.create_xml_infos(df)

        StreamlitStateManager.set("data", data)

        based_prompt = f"""Act as a data analyst specializing in employee feedback and customer request management. Your objective is to create a comprehensive report that evaluates the current state of global duty-free operations, leveraging structured data from employee feedback and customer-related comments. Your task is to **present granular, concrete data points** without overly summarizing or generalizing the information. Don't give Impact analysis or Recommendations unless specifically asked for.

            ### <task overview>
            You are tasked to analyze the provided data and generate a detailed report that addresses the following key aspects:

            #### **Top 5 Global Operational Challenges**
            - Identify and list the **top issues** affecting global duty-free operations. Provide the issues in **specific, concrete terms**, highlighting granular data points.
            - Present direct data points (e.g., specific SKUs, prices, promotions, or durations of out-of-stock situations).
            - Avoid summarizing; instead, provide precise observations drawn from the dataset.
            - Categorize these issues into specific areas (e.g., inventory management, service quality, customer behavior).
            - Support findings with **exact examples from the data**, citing relevant Field Intelligence and Row_ID values (as many as possible). Be very detailed.

            ### **<instructions>**
            The report must include **Specific Examples**: For each problem, provide exact product names, SKUs, promotional details, or other concrete data points (e.g., "New SKU from Pueblo cigarettes Blue King Size RSP at AED147" or "WILL's promotion offers a 15% discount on 3 packs, RSP AED 46 each").
            {GLOBAL_CHALLENGES_FORMAT}


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
            {data}
            </data>

            Lets think step by step.
            """

        return based_prompt

    @classmethod
    def create_xml_infos(cls, df):
        """
        Constructs an XML structure from a filtered DataFrame for generating prompt data.

        Parameters:
        -----------
        df : pandas.DataFrame
            A DataFrame that has been pre-filtered to contain only relevant samples.
            This DataFrame should already have the necessary filtering applied.

        Returns:
        --------
        str
            An XML string representing the data.
        """

        # Add Row_ID to the DataFrame
        df["row_ID"] = df.index

        # Required columns (case-insensitive)
        required_columns = [
            "field_intelligence_translated",
            "submission_datetime",
            "class",
            "location_name",
            "form_type",
            "product_category_name",
            "brand_name_from",
            "brand_name_to",
            "pmi_product_name",
            "tmo_name",
            "row_id",
        ]

        # Normalize column names in the DataFrame to lowercase for case-insensitive matching
        df.columns = [col.lower() for col in df.columns]
        # Ensure all required columns exist in the DataFrame
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f"Missing required columns in DataFrame: {missing_columns}"
            )

        # Create the root element of the XML
        root = ET.Element("Entries")

        for _, row in df.iterrows():
            # Create an entry element
            entry = ET.SubElement(root, "Entry")

            # Add the Row ID
            ET.SubElement(entry, "Row_ID").text = str(row["row_id"])

            ET.SubElement(entry, "Class").text = str(row["class"])
            ET.SubElement(entry, "Field_Intelligence_Translated").text = str(
                row["field_intelligence_translated"]
            )
            ET.SubElement(entry, "Submission Date").text = str(
                row["submission_datetime"]
            )

            # Add optional fields if they exist
            if pd.notna(row["location_name"]):
                ET.SubElement(entry, "Location").text = str(row["location_name"])

            # Add Form_Type and nested details
            form_type = ET.SubElement(entry, "Form_Type")
            form_type.text = str(row["form_type"])

            form_type_details = ET.SubElement(entry, "Form_Type_Details")
            for col in [
                "product_category_name",
                "brand_name_from",
                "brand_name_to",
                "pmi_product_name",
                "tmo_name",
            ]:
                if pd.notna(row[col]):
                    ET.SubElement(form_type_details, col).text = str(row[col])

        # Convert the tree to a string
        return ET.tostring(root, encoding="unicode", method="xml")

    @classmethod
    def extract_report_from_output(cls, output: str):
        """
        Extracts, cleans, and parses the XML content from a given string.

        Parameters:
            output (str): The string containing XML content.

        Returns:
            str: The content of the <report> tag, or an error message if the XML cannot be processed.
        """
        # Step 1: Clean the string to isolate the XML content
        start = output.find("<response>")  # Find the start of the XML
        if start == -1:
            return "No XML content found in the output."

        xml_content = output[start:]  # Extract the XML part

        # Step 2: Escape problematic characters in <report> content
        xml_content = re.sub(
            r"<report>(.*?)</report>",
            lambda m: f"<report><![CDATA[{m.group(1).strip()}]]></report>",
            xml_content,
            flags=re.DOTALL,
        )

        # Step 3: Parse the cleaned XML
        try:
            root = ET.fromstring(xml_content)
            # Extract the <report> content
            report = root.find("report")
            return (
                report.text.strip()
                if report is not None
                else "No <report> tag found in the XML."
            )
        except ET.ParseError as e:
            return f"Error parsing XML: {e}\nCleaned XML string for debugging:\n{xml_content}"

    @classmethod
    def extract_examples_id_from_output(cls, output: str):
        """
        Extracts all the information between <examplesID> tags from a given string.

        Parameters:
            output (str): The string containing mixed XML and non-XML content.

        Returns:
            list: A list of strings representing the content inside <examplesID> tags,
                  or an error message if the XML cannot be processed.
        """
        # Step 1: Isolate XML content between <examplesID> tags using regex
        pattern = r"<examplesID>(.*?)</examplesID>"
        matches = re.findall(pattern, output, flags=re.DOTALL)

        if not matches:
            return "No <examplesID> tags found in the output."

        results = []
        for match in matches:
            # Split content by commas or newlines and clean each item
            items = re.split(r"[,\n]", match)
            results.extend(item.strip() for item in items if item.strip())

        return results

    @staticmethod
    def generate_report(base_matrix):
        """
        Generates a report based on the given base matrix and updates the session state.

        Parameters:
            base_matrix: The data used to create the report prompt.

        Returns:
            None
        """
        max_attempts = 3
        attempt = 0
        while attempt < max_attempts:
            try:
                # Create the report prompt
                prompt = ReportGeneration.create_generic_prompt_report(base_matrix)
                StreamlitStateManager.set("prompt", prompt)

                # Initialize response generator
                response_generator = ReportGeneration.api_chat_completions_stream(
                    prompt
                )

                # Show spinner and handle the response
                response_or_func_call, prob = stream_message_from_anthropic(
                    response_generator
                )
                StreamlitStateManager.set("generator_output", response_or_func_call)

                # Mark the report as generated
                StreamlitStateManager.set("report_generated", True)
                return

            except Exception as e:
                attempt += 1
                st.warning(
                    f"Error: {e}\nRetrying attempt {attempt} of {max_attempts}..."
                )
                if attempt >= max_attempts:
                    st.warning(
                        "Connection couldn't be established after multiple attempts. Consider reloading the page."
                    )
                    st.error(f"Error: {e}")
                    st.stop()

    @staticmethod
    def extract_infos_from_report(output):
        try:
            report = ReportGeneration.extract_report_from_output(output)
            StreamlitStateManager.set("report", report)

            examplesID = ReportGeneration.extract_examples_id_from_output(output)
            StreamlitStateManager.set("examplesID", examplesID)
        except Exception as e:
            st.error(f"Error Extracting the report or the examples: {e}")
            st.stop()


class ReportCombination(ReportGeneration, ConnectionAPI):

    @classmethod
    def determine_intermediate_reports(
        cls, base_matrix: pd.DataFrame, filters: pd.DataFrame
    ) -> list[pd.DataFrame]:
        """
        Determines the number of intermediate reports to generate by splitting the `base_matrix`
        into sub-groups based on the filtering criteria.

        The function follows these rules:
        - If the date range in `filters` is 1 month or less, the base matrix is grouped by 'VP_REGION_NAME',
        ensuring each subgroup has at most 1200 rows.
        - If the date range is more than 1 month, it is grouped by 'SUBMISSION_DATETIME' with a max of 1200 rows per group.

        Args:
            base_matrix (pd.DataFrame): The dataset containing the raw report data.
            filters (pd.DataFrame): The filtering criteria, expected to contain a 'DATE_RANGE' column.

        Returns:
            list[pd.DataFrame]: A list of DataFrames, each representing a sub-group for report generation.
        """

        # Rename filters if necessary
        filters = BaseMatrixService.rename_scope(filters)

        # Extract and parse start and end dates from DATE_RANGE
        date_range_str = filters["DATE_RANGE"].iloc[0]
        start_date_str, end_date_str = map(str.strip, date_range_str.split("—"))
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

        # Calculate the month difference
        month_diff = (end_date.year - start_date.year) * 12 + (
            end_date.month - start_date.month
        )

        # Determine the grouping column based on the date range length
        grouping_column = "VP_REGION_NAME" if month_diff < 1 else "SUBMISSION_DATETIME"

        # Define max rows per subgroup
        max_rows = 1200

        # Sort the DataFrame by the grouping column
        base_matrix = base_matrix.sort_values(by=grouping_column)

        # Calculate the number of required splits
        num_splits = -(
            -len(base_matrix) // max_rows
        )  # Equivalent to ceil(len(base_matrix) / max_rows)

        # Split DataFrame into evenly sized chunks
        sub_groups = np.array_split(base_matrix, num_splits)

        StreamlitStateManager.set("start_generation", True)
        StreamlitStateManager.set("subgroups", sub_groups)
        return sub_groups

    @classmethod
    def create_combination_prompt(cls, reports: list[str]) -> str:
        """Create Prompt by formatting a list of reports as strings into a structured representation."""

        reports_str = "\n\n".join(
            f"Report {i + 1}:\n{report}" for i, report in enumerate(reports)
        )

        based_prompt = f"""Act as a data analyst specializing in employee feedback and customer request management. Your objective is to create a comprehensive report that evaluates the current state of global duty-free operations, leveraging structured data from employee feedback and customer-related comments. Your task is to **present granular, concrete data points** without overly summarizing or generalizing the information. Don't give Impact analysis or Recommendations unless specifically asked for.

            ### <task overview>
            You are tasked to analyze the provided data and generate a detailed report that addresses the following key aspects:
            Intermediates report have been generated and now you need to combine them into a single report.

            #### **Top 5 Global Operational Challenges**
            - Identify and list the **top issues** affecting global duty-free operations. Provide the issues in **specific, concrete terms**, highlighting granular data points.
            - Present direct data points (e.g., specific SKUs, prices, promotions, or durations of out-of-stock situations).
            - Avoid summarizing; instead, provide precise observations drawn from the dataset.
            - Categorize these issues into specific areas (e.g., inventory management, service quality, customer behavior).
            - Support findings with **exact examples from the data**, citing relevant Field Intelligence and Row_ID values (as many as possible). Be very detailed.

            ### **<instructions>**
            The report must include **Specific Examples**: For each problem, provide exact product names, SKUs, promotional details, or other concrete data points (e.g., "New SKU from Pueblo cigarettes Blue King Size RSP at AED147" or "WILL's promotion offers a 15% discount on 3 packs, RSP AED 46 each").
            {GLOBAL_CHALLENGES_FORMAT}

            <output format>:
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

            <formatting instructions>:
            - List specific examples for each insight, such as SKUs, promotions, or exact timelines.
            - Emphasize key details like product names, exact prices, or promotion terms. Avoid vague phrases. What we don't want: "Some products were out of stock.", "Many customers complained about pricing.". What we want: "Product X (SKU: 12345) was out of stock for 3 days.", "Customers reported a 20% price increase for Brand Y in the last month."
            - Include tables with product names and details, if applicable.
            </formatting instructions>

            Here are the intermediate reports:
            <reports>
            {reports_str}
            </reports>

            Let's think step by step.
            """

        return based_prompt

    @classmethod
    def generate_intermediate_report(cls, base_matrix) -> str:
        """
        Generates a report based on the given base matrix and updates the session state.

        Parameters:
            base_matrix: The data used to create the report prompt.

        Returns:
            None
        """
        max_attempts = 3
        attempt = 0
        while attempt < max_attempts:
            try:
                # Create the report prompt
                prompt = ReportGeneration.create_generic_prompt_report(base_matrix)
                StreamlitStateManager.set("prompt", prompt)

                # Generation of the response
                response = ReportGeneration.api_chat_completions(prompt)
                intermediate_report = response["output"]["message"]["content"][0][
                    "text"
                ]
                return intermediate_report

            except Exception as e:
                attempt += 1
                st.warning(
                    f"Error: {e}\nRetrying attempt {attempt} of {max_attempts}..."
                )
                if attempt >= max_attempts:
                    st.warning(
                        "Connection couldn't be established after multiple attempts. Consider reloading the page."
                    )
                    st.error(f"Error: {e}")
                    st.stop()

    @classmethod
    def combine_reports(cls, reports: list) -> None:
        """
        Combines the intermediate reports into a single report and updates the session state.

        Returns:
            None
        """

        # Get prompt
        max_attempts = 3
        attempt = 0
        while attempt < max_attempts:
            try:
                # Create the report prompt
                prompt = ReportCombination.create_combination_prompt(reports)
                StreamlitStateManager.set("prompt", prompt)

                # Initialize response generator
                response_generator = ReportGeneration.api_chat_completions_stream(
                    prompt
                )

                # Show spinner and handle the response
                response_or_func_call, prob = stream_message_from_anthropic(
                    response_generator
                )
                StreamlitStateManager.set("generator_output", response_or_func_call)

                # Mark the report as generated
                StreamlitStateManager.set("report_generated", True)
                return None

            except Exception as e:
                attempt += 1
                st.warning(
                    f"Error: {e}\nRetrying attempt {attempt} of {max_attempts}..."
                )
                if attempt >= max_attempts:
                    st.warning(
                        "Connection couldn't be established after multiple attempts. Consider reloading the page."
                    )
                    st.error(f"Error: {e}")
                    st.stop()