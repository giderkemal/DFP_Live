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
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from enum import Enum

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


class ReportSection(Enum):
    """Enumeration of available report sections"""
    GLOBAL_CHALLENGES = "global_challenges"
    LOCATION_BREAKDOWN = "location_breakdown"
    TIMING_TRENDS = "timing_trends"
    EXECUTIVE_SUMMARY = "executive_summary"


@dataclass
class ReportTemplate:
    """Configuration for report template sections"""
    name: str
    description: str
    sections: List[ReportSection]
    max_challenges: int = 5
    require_citations: bool = True
    include_recommendations: bool = False
    output_format: str = "xml"
    
    def validate(self) -> tuple[bool, Optional[str]]:
        """Validate template configuration"""
        if not self.name:
            return False, "Template name cannot be empty"
        if not self.sections:
            return False, "Template must have at least one section"
        if self.max_challenges < 1 or self.max_challenges > 10:
            return False, "Max challenges must be between 1 and 10"
        return True, None


class ReportTemplateManager:
    """Manages report templates and prompt generation with robust error handling"""
    
    # Default templates
    DEFAULT_TEMPLATES = {
        "generic": ReportTemplate(
            name="Generic Analysis Report",
            description="Comprehensive analysis focusing on specific data points",
            sections=[ReportSection.GLOBAL_CHALLENGES],
            max_challenges=5,
            require_citations=True,
            include_recommendations=False
        ),
        "comprehensive": ReportTemplate(
            name="Comprehensive Analysis Report", 
            description="Full analysis with location and timing insights",
            sections=[
                ReportSection.GLOBAL_CHALLENGES,
                ReportSection.LOCATION_BREAKDOWN,
                ReportSection.TIMING_TRENDS
            ],
            max_challenges=5,
            require_citations=True,
            include_recommendations=True
        )
    }
    
    @classmethod
    def get_template(cls, template_name: str = "generic") -> ReportTemplate:
        """Get a template by name with fallback to default"""
        template = cls.DEFAULT_TEMPLATES.get(template_name, cls.DEFAULT_TEMPLATES["generic"])
        is_valid, error = template.validate()
        if not is_valid:
            logger.warning(f"Template validation failed: {error}. Using default template.")
            return cls.DEFAULT_TEMPLATES["generic"]
        return template
    
    @classmethod
    def build_section_content(cls, section: ReportSection, template: ReportTemplate) -> str:
        """Build content for a specific report section"""
        
        if section == ReportSection.GLOBAL_CHALLENGES:
            citations = """
Support insights with multiple examples (as many as possible, minimum 5), citing their Row_ID in this format:
[Row_ID:row_id]
Example Citation Format:
- [Row_ID:80]""" if template.require_citations else ""
            
            return f"""
            #### **Top {template.max_challenges} Global Operational Challenges**
            - Identify and list the **top issues** affecting global duty-free operations. Provide the issues in **specific, concrete terms**, highlighting granular data points.
            - Present direct data points (e.g., specific SKUs, prices, promotions, or durations of out-of-stock situations).
            - Avoid summarizing; instead, provide precise observations drawn from the dataset.
            - Categorize these issues into specific areas (e.g., inventory management, service quality, customer behavior).
            - Support findings with **exact examples from the data**, citing relevant Field Intelligence and Row_ID values (as many as possible). Be very detailed.
            {citations}
            """
            
        elif section == ReportSection.LOCATION_BREAKDOWN:
            return """
            #### **Location-Specific Analysis**
            - Break down findings by region or specific duty-free stations.
            - Highlight patterns or recurrent issues in different locations.
            - Support observations with specific data examples (Field Intelligence and Row_ID).
            """
            
        elif section == ReportSection.TIMING_TRENDS:
            return """
            #### **Timing and Seasonal Trends**
            - Analyze feedback for patterns tied to specific times of the year.
            - Identify spikes in issues or unique challenges during these periods.
            - Provide examples illustrating these trends.
            """
            
        return ""
    
    @classmethod
    def create_structured_prompt(
        cls,
        df: pd.DataFrame,
        template_name: str = "generic",
        extra_demand: str = "",
        extra_instructions: str = ""
    ) -> tuple[str, bool]:
        """
        Create a structured prompt with comprehensive error handling
        
        Returns:
            tuple: (prompt_string, success_flag)
        """
        try:
            # Get and validate template
            template = cls.get_template(template_name)
            
            # Generate XML data with error handling
            try:
                data = XMLDataProcessor.create_xml_infos(df)
                StreamlitStateManager.set("data", data)
            except Exception as e:
                logger.error(f"Failed to create XML data: {e}")
                return f"Error: Failed to process data - {str(e)}", False
            
            # Build sections content
            sections_content = ""
            for section in template.sections:
                sections_content += cls.build_section_content(section, template)
            
            # Add extra content if provided
            if extra_demand:
                sections_content += f"\n\n#### **Additional Requirements**\n{extra_demand}"
            
            # Build instructions
            instructions = f"""
            ### **<instructions>**
            The report must include **Specific Examples**: For each problem, provide exact product names, SKUs, promotional details, or other concrete data points (e.g., "New SKU from Pueblo cigarettes Blue King Size RSP at AED147" or "WILL's promotion offers a 15% discount on 3 packs, RSP AED 46 each").
            
            {extra_instructions}
            
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
            """
            
            # Build the complete prompt
            prompt = f"""Act as a data analyst specializing in employee feedback and customer request management. Your objective is to create a comprehensive report that evaluates the current state of global duty-free operations, leveraging structured data from employee feedback and customer-related comments. Your task is to **present granular, concrete data points** without overly summarizing or generalizing the information. {"Don't give Impact analysis or Recommendations unless specifically asked for." if not template.include_recommendations else "Include actionable recommendations for each insight."}

            ### <task overview>
            You are tasked to analyze the provided data and generate a detailed report that addresses the following key aspects:
            {sections_content}
            
            {instructions}

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

            Let's think step by step.
            """
            
            return prompt, True
            
        except Exception as e:
            logger.error(f"Failed to create structured prompt: {e}")
            return f"Error: Failed to create prompt - {str(e)}", False


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
        """
        Create Prompt using the new ReportTemplateManager system for better structure and error handling.
        
        This method now serves as a wrapper for the improved template system while maintaining 
        backward compatibility.
        """
        try:
            # Use the new template manager for creating structured prompts
            prompt, success = ReportTemplateManager.create_structured_prompt(
                df=df,
                template_name="generic",
                extra_demand=EXTRA_DEMAND,
                extra_instructions=EXTRA_INSTRUCTIONS
            )
            
            if success:
                return prompt
            else:
                # Log the issue but try to continue with a fallback
                logger.warning(f"Template manager failed, using fallback: {prompt}")
                # Fall back to a simple prompt
                return cls._create_fallback_prompt(df)
                
        except Exception as e:
            logger.error(f"Failed to create prompt with template manager: {e}")
            # Final fallback
            return cls._create_fallback_prompt(df)
    
    @classmethod
    def _create_fallback_prompt(cls, df):
        """Fallback prompt creation method"""
        try:
            data = XMLDataProcessor.create_xml_infos(df)
            return f"""
            Act as a data analyst. Analyze the provided data and create a report focusing on:
            1. Top 5 operational challenges with specific examples
            2. Cite Row_ID references in format [Row_ID:123]
            
            Output in XML format:
            <response>
                <report>Your analysis here</report>
                <examplesID></examplesID>
            </response>
            
            Data:
            <data>{data}</data>
            """
        except Exception as e:
            logger.error(f"Even fallback prompt creation failed: {e}")
            return """
            <response>
                <report>Error: Unable to process data for report generation.</report>
                <examplesID></examplesID>
            </response>
            """

    @classmethod
    def create_xml_infos(cls, df):
        """
        Constructs an XML structure from a filtered DataFrame using the enhanced XMLDataProcessor.
        
        This method now serves as a wrapper for the improved XML processing system.
        """
        return XMLDataProcessor.create_xml_infos(df)

    @staticmethod
    def generate_report(base_matrix):
        """
        Enhanced report generation with comprehensive error handling and graceful degradation.

        Parameters:
            base_matrix: The data used to create the report prompt.

        Returns:
            None
        """
        max_attempts = 3
        attempt = 0
        
        # Validate input data
        if base_matrix is None:
            st.error("No data provided for report generation.")
            return
            
        if hasattr(base_matrix, 'empty') and base_matrix.empty:
            st.error("No data available for report generation. Please adjust your filters.")
            return
        
        while attempt < max_attempts:
            try:
                # Create the report prompt with enhanced error handling
                prompt = ReportGeneration.create_generic_prompt_report(base_matrix)
                
                # Check if prompt creation was successful
                if prompt.startswith("Error:"):
                    raise ValueError(f"Prompt creation failed: {prompt}")
                    
                StreamlitStateManager.set("prompt", prompt)

                # Initialize response generator with validation
                response_generator = ReportGeneration.api_chat_completions_stream(prompt)
                
                if response_generator is None:
                    raise ValueError("API response generator is None")

                # Show spinner and handle the response
                response_or_func_call, prob = stream_message_from_anthropic(response_generator)
                
                # Validate response
                if not response_or_func_call:
                    raise ValueError("Empty response from API")
                    
                if response_or_func_call.startswith("Error:"):
                    raise ValueError(f"API error: {response_or_func_call}")

                StreamlitStateManager.set("generator_output", response_or_func_call)

                # Mark the report as generated
                StreamlitStateManager.set("report_generated", True)
                return

            except Exception as e:
                attempt += 1
                error_msg = str(e)
                logger.error(f"Report generation attempt {attempt} failed: {error_msg}")
                
                if attempt < max_attempts:
                    st.warning(f"Attempt {attempt} failed: {error_msg}. Retrying...")
                    time.sleep(2)  # Brief delay before retry
                else:
                    # Final attempt failed - provide graceful degradation
                    st.error("Report generation failed after multiple attempts.")
                    
                    # Try to provide a basic summary instead of crashing
                    try:
                        basic_summary = ReportGeneration._create_basic_summary(base_matrix)
                        StreamlitStateManager.set("generator_output", basic_summary)
                        StreamlitStateManager.set("report_generated", True)
                        st.info("Provided a basic data summary instead of full report.")
                        return
                    except Exception as summary_error:
                        logger.error(f"Even basic summary creation failed: {summary_error}")
                        st.error("Unable to generate any report. Please check your data and try again.")
                        return

    @staticmethod
    def _create_basic_summary(base_matrix):
        """Create a basic summary when full report generation fails"""
        try:
            row_count = len(base_matrix)
            
            # Get basic statistics
            if 'class' in base_matrix.columns:
                top_classes = base_matrix['class'].value_counts().head(3)
                class_summary = "\n".join([f"- {cls}: {count} occurrences" for cls, count in top_classes.items()])
            else:
                class_summary = "Class information not available"
            
            if 'location_name' in base_matrix.columns:
                top_locations = base_matrix['location_name'].value_counts().head(3)
                location_summary = "\n".join([f"- {loc}: {count} occurrences" for loc, count in top_locations.items()])
            else:
                location_summary = "Location information not available"
            
            basic_report = f"""
            <response>
                <report>
                Data Summary Report
                
                Total Records: {row_count}
                
                Top Issue Categories:
                {class_summary}
                
                Top Locations:
                {location_summary}
                
                Note: This is a basic summary generated due to technical issues with full report generation.
                </report>
                <examplesID></examplesID>
            </response>
            """
            
            return basic_report
            
        except Exception as e:
            logger.error(f"Basic summary creation failed: {e}")
            return """
            <response>
                <report>Error: Unable to process data for report generation.</report>
                <examplesID></examplesID>
            </response>
            """

    @staticmethod
    def extract_infos_from_report(output):
        """Enhanced report extraction with comprehensive error handling"""
        try:
            if not output:
                st.error("No output provided for report extraction.")
                return
            
            # Use the new robust extractor
            report, report_success = ReportExtractor.extract_report_from_output(output)
            examples_ids, ids_success = ReportExtractor.extract_examples_id_from_output(output)
            
            # Set the extracted information with success indicators
            StreamlitStateManager.set("report", report)
            StreamlitStateManager.set("examplesID", examples_ids)
            
            # Log extraction results
            if not report_success:
                logger.warning("Report extraction used fallback method")
                st.info("Report was extracted using fallback method - formatting may be simplified.")
                
            if not ids_success and examples_ids:
                logger.warning("Example IDs extraction used fallback method")
                st.info("Example references were found using alternative extraction.")
            elif not examples_ids:
                logger.info("No example IDs found in the report")
                
        except Exception as e:
            logger.error(f"Error extracting report information: {e}")
            st.error(f"Error processing report: {str(e)}")
            
            # Set fallback values instead of stopping the app
            StreamlitStateManager.set("report", "Error: Unable to extract report content.")
            StreamlitStateManager.set("examplesID", [])


class ReportExtractor:
    """Robust report extraction with fallback mechanisms"""
    
    @staticmethod
    def extract_report_from_output(output: str) -> tuple[str, bool]:
        """
        Enhanced report extraction with multiple fallback strategies

        Returns:
            tuple: (extracted_report, success_flag)
        """
        if not output or not isinstance(output, str):
            return "Error: No valid output provided", False
        
        # Strategy 1: Try XML parsing with CDATA handling
        try:
            start = output.find("<response>")
            if start != -1:
                xml_content = output[start:]
                
                # Escape problematic characters in <report> content
        xml_content = re.sub(
            r"<report>(.*?)</report>",
            lambda m: f"<report><![CDATA[{m.group(1).strip()}]]></report>",
            xml_content,
            flags=re.DOTALL,
        )

            root = ET.fromstring(xml_content)
            report = root.find("report")
                if report is not None and report.text:
                    return report.text.strip(), True
        except ET.ParseError as e:
            logger.warning(f"XML parsing failed: {e}")
        except Exception as e:
            logger.warning(f"XML extraction failed: {e}")
        
        # Strategy 2: Try simple regex extraction
        try:
            # Look for content between <report> tags without XML parsing
            pattern = r"<report>(.*?)</report>"
            match = re.search(pattern, output, flags=re.DOTALL)
            if match:
                content = match.group(1).strip()
                # Remove CDATA wrapper if present
                content = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", content, flags=re.DOTALL)
                if content:
                    return content, True
        except Exception as e:
            logger.warning(f"Regex extraction failed: {e}")
        
        # Strategy 3: Look for structured content patterns
        try:
            # Look for common report headers/patterns
            patterns = [
                r"(?:Top \d+ (?:Global )?(?:Operational )?Challenges?|Analysis|Insights?|Summary).*?(?=\n\n|\Z)",
                r"(?:Problem|Issue|Challenge) \d+:.*?(?=(?:Problem|Issue|Challenge) \d+:|\Z)",
                r"\d+\.\s+.*?(?=\n\d+\.|\Z)"
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, output, flags=re.DOTALL | re.IGNORECASE)
                if matches:
                    structured_content = "\n\n".join(matches)
                    if len(structured_content.strip()) > 100:  # Minimum content length
                        return structured_content.strip(), True
        except Exception as e:
            logger.warning(f"Pattern extraction failed: {e}")
        
        # Strategy 4: Return trimmed output as fallback
        try:
            # Remove XML tags and return cleaned content
            cleaned = re.sub(r"<[^>]+>", "", output)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            if len(cleaned) > 50:  # Minimum content threshold
                return cleaned, False  # Flag as fallback
        except Exception as e:
            logger.warning(f"Fallback cleaning failed: {e}")
        
        return "Error: Unable to extract report content from response", False

    @staticmethod
    def extract_examples_id_from_output(output: str) -> tuple[List[str], bool]:
        """
        Enhanced example ID extraction with multiple strategies

        Returns:
            tuple: (list_of_ids, success_flag)
        """
        if not output or not isinstance(output, str):
            return [], False
        
        # Strategy 1: XML parsing
        try:
            start = output.find("<response>")
            if start != -1:
                xml_content = output[start:]
                root = ET.fromstring(xml_content)
                examples_elem = root.find("examplesID")
                if examples_elem is not None and examples_elem.text:
                    # Split and clean IDs
                    ids = re.split(r"[,\n\s]+", examples_elem.text.strip())
                    cleaned_ids = [id.strip() for id in ids if id.strip()]
                    return cleaned_ids, True
        except Exception as e:
            logger.warning(f"XML ID extraction failed: {e}")
        
        # Strategy 2: Regex extraction from XML tags
        try:
            pattern = r"<examplesID>(.*?)</examplesID>"
            matches = re.findall(pattern, output, flags=re.DOTALL)
            if matches:
                all_ids = []
                for match in matches:
                    ids = re.split(r"[,\n\s]+", match.strip())
                    all_ids.extend([id.strip() for id in ids if id.strip()])
                return all_ids, True
        except Exception as e:
            logger.warning(f"Regex ID extraction failed: {e}")
        
        # Strategy 3: Look for Row_ID patterns in text
        try:
            # Find all [Row_ID:123] patterns
            pattern = r"\[Row_ID:(\d+)\]"
            matches = re.findall(pattern, output)
            if matches:
                return list(set(matches)), True  # Remove duplicates
            except Exception as e:
            logger.warning(f"Pattern ID extraction failed: {e}")
        
        # Strategy 4: Look for any numeric references that might be IDs
        try:
            # Look for patterns like "row 123", "ID 123", etc.
            patterns = [
                r"(?:row|id|example)\s*:?\s*(\d+)",
                r"(?:Row_ID|rowid|row_id)\s*:?\s*(\d+)"
            ]
            
            found_ids = []
            for pattern in patterns:
                matches = re.findall(pattern, output, flags=re.IGNORECASE)
                found_ids.extend(matches)
            
            if found_ids:
                return list(set(found_ids)), False  # Flag as fallback
        except Exception as e:
            logger.warning(f"Fallback ID extraction failed: {e}")
        
        return [], False


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
            
            Support insights with multiple examples, citing their Row_ID in this format:
            [Row_ID:row_id]
            Example Citation Format:
            - [Row_ID:80]

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
        Enhanced intermediate report generation with robust error handling.

        Parameters:
            base_matrix: The data used to create the report prompt.

        Returns:
            str: The generated intermediate report content or a fallback summary
        """
        max_attempts = 3
        attempt = 0
        
        # Validate input
        if base_matrix is None or (hasattr(base_matrix, 'empty') and base_matrix.empty):
            logger.warning("Empty or None base_matrix provided for intermediate report")
            return cls._create_fallback_intermediate_report()
        
        while attempt < max_attempts:
            try:
                # Create the report prompt with validation
                prompt = ReportGeneration.create_generic_prompt_report(base_matrix)
                
                if prompt.startswith("Error:"):
                    raise ValueError(f"Prompt creation failed: {prompt}")
                    
                StreamlitStateManager.set("prompt", prompt)

                # Generation of the response with validation
                response = ReportGeneration.api_chat_completions(prompt)
                
                if not response or "error" in response:
                    raise ValueError(f"API response error: {response}")
                
                # Extract the report content safely
                try:
                    intermediate_report = response["output"]["message"]["content"][0]["text"]
                    if intermediate_report and len(intermediate_report.strip()) > 10:
                return intermediate_report
                    else:
                        raise ValueError("Empty or too short response content")
                except (KeyError, IndexError, TypeError) as e:
                    raise ValueError(f"Invalid response structure: {e}")

            except Exception as e:
                attempt += 1
                error_msg = str(e)
                logger.error(f"Intermediate report generation attempt {attempt} failed: {error_msg}")
                
                if attempt < max_attempts:
                    logger.info(f"Retrying intermediate report generation... ({attempt}/{max_attempts})")
                    time.sleep(1)  # Brief delay before retry
                else:
                    logger.error("All attempts failed for intermediate report generation")
                    # Return a fallback summary instead of crashing
                    return cls._create_fallback_intermediate_report(base_matrix, error_msg)
    
    @classmethod
    def _create_fallback_intermediate_report(cls, base_matrix=None, error_msg=""):
        """Create a basic fallback report when intermediate generation fails"""
        try:
            if base_matrix is not None and not base_matrix.empty:
                row_count = len(base_matrix)
                basic_content = f"Partial analysis of {row_count} records due to processing issues."
            else:
                basic_content = "Unable to process data segment."
            
            return f"""
            <response>
                <report>
                Intermediate Report (Fallback)
                
                {basic_content}
                
                Note: This is a simplified summary due to technical issues during report generation.
                Error: {error_msg}
                </report>
                <examplesID></examplesID>
            </response>
            """
        except Exception as e:
            logger.error(f"Even fallback intermediate report creation failed: {e}")
            return """
            <response>
                <report>Error: Unable to generate intermediate report.</report>
                <examplesID></examplesID>
            </response>
            """

    @classmethod
    def combine_reports(cls, reports: list) -> None:
        """
        Enhanced report combination with robust error handling and graceful degradation.

        Parameters:
            reports: List of intermediate report strings to combine

        Returns:
            None
        """

        # Validate input
        if not reports:
            st.error("No intermediate reports provided for combination.")
            return
        
        # Filter out empty or error reports
        valid_reports = [r for r in reports if r and not r.strip().startswith("Error:")]
        
        if not valid_reports:
            st.error("No valid intermediate reports available for combination.")
            # Create a basic combined summary instead
            cls._create_fallback_combined_report(reports)
            return
        
        if len(valid_reports) < len(reports):
            st.warning(f"Some intermediate reports had issues. Combining {len(valid_reports)} out of {len(reports)} reports.")

        max_attempts = 3
        attempt = 0
        
        while attempt < max_attempts:
            try:
                # Create the report prompt with validation
                prompt = ReportCombination.create_combination_prompt(valid_reports)
                
                if not prompt or len(prompt.strip()) < 100:
                    raise ValueError("Generated prompt is too short or empty")
                
                StreamlitStateManager.set("prompt", prompt)

                # Initialize response generator with validation
                response_generator = ReportGeneration.api_chat_completions_stream(prompt)
                
                if response_generator is None:
                    raise ValueError("API response generator is None")

                # Show spinner and handle the response
                response_or_func_call, prob = stream_message_from_anthropic(response_generator)
                
                # Validate response
                if not response_or_func_call:
                    raise ValueError("Empty response from API")
                    
                if response_or_func_call.startswith("Error:"):
                    raise ValueError(f"API error: {response_or_func_call}")

                StreamlitStateManager.set("generator_output", response_or_func_call)

                # Mark the report as generated
                StreamlitStateManager.set("report_generated", True)
                return

            except Exception as e:
                attempt += 1
                error_msg = str(e)
                logger.error(f"Report combination attempt {attempt} failed: {error_msg}")
                
                if attempt < max_attempts:
                    st.warning(f"Combination attempt {attempt} failed. Retrying...")
                    time.sleep(2)  # Brief delay before retry
                else:
                    # Final attempt failed - provide graceful degradation
                    st.error("Report combination failed after multiple attempts.")
                    
                    # Try to create a basic combined summary instead
                    try:
                        cls._create_fallback_combined_report(valid_reports)
                        st.info("Created a basic combined summary instead of full report.")
                        return
                    except Exception as fallback_error:
                        logger.error(f"Even fallback combination failed: {fallback_error}")
                        st.error("Unable to combine reports. Please try again later.")
                        return
    
    @classmethod
    def _create_fallback_combined_report(cls, reports: list):
        """Create a basic combined report when advanced combination fails"""
        try:
            report_count = len(reports)
            
            # Extract key information from each report
            combined_content = f"Combined Analysis Summary\n\nTotal Segments Analyzed: {report_count}\n\n"
            
            for i, report in enumerate(reports[:3], 1):  # Limit to first 3 for brevity
                # Try to extract key content
                if "<report>" in report:
                    content_match = re.search(r"<report>(.*?)</report>", report, re.DOTALL)
                    if content_match:
                        content = content_match.group(1).strip()
                        # Take first 200 characters
                        summary = content[:200] + "..." if len(content) > 200 else content
                        combined_content += f"Segment {i} Summary:\n{summary}\n\n"
            
            if report_count > 3:
                combined_content += f"...and {report_count - 3} additional segments analyzed.\n\n"
            
            combined_content += "Note: This is a simplified combination due to technical issues with advanced report merging."
            
            fallback_report = f"""
            <response>
                <report>
                {combined_content}
                </report>
                <examplesID></examplesID>
            </response>
            """
            
            StreamlitStateManager.set("generator_output", fallback_report)
            StreamlitStateManager.set("report_generated", True)
            
        except Exception as e:
            logger.error(f"Fallback combination creation failed: {e}")
            # Set minimal error report
            error_report = """
            <response>
                <report>Error: Unable to combine intermediate reports.</report>
                <examplesID></examplesID>
            </response>
            """
            StreamlitStateManager.set("generator_output", error_report)
            StreamlitStateManager.set("report_generated", True)


class XMLDataProcessor:
    """Utility class for XML data processing operations"""
    
    @staticmethod
    def create_xml_infos(df):
        """
        Constructs an XML structure from a filtered DataFrame with enhanced error handling.

        Parameters:
        -----------
        df : pandas.DataFrame
            A DataFrame that has been pre-filtered to contain only relevant samples.

        Returns:
        --------
        str
            An XML string representing the data.
        """
        try:
            if df is None or df.empty:
                raise ValueError("DataFrame is None or empty")

            # Create a copy to avoid modifying the original
            df_copy = df.copy()
            
            # Add Row_ID to the DataFrame
            df_copy["row_ID"] = df_copy.index

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
            df_copy.columns = [col.lower() for col in df_copy.columns]
            
            # Ensure all required columns exist in the DataFrame
            missing_columns = [col for col in required_columns if col not in df_copy.columns]
            if missing_columns:
                # Instead of raising error, create missing columns with default values
                logger.warning(f"Missing columns will be filled with defaults: {missing_columns}")
                for col in missing_columns:
                    df_copy[col] = "N/A"

            # Create the root element of the XML
            root = ET.Element("Entries")

            for idx, row in df_copy.iterrows():
                try:
                    # Create an entry element
                    entry = ET.SubElement(root, "Entry")

                    # Add the Row ID with validation
                    row_id_elem = ET.SubElement(entry, "Row_ID")
                    row_id_elem.text = XMLDataProcessor._sanitize_xml_text(str(row.get("row_id", idx)))

                    # Add required fields with safe access and validation
                    class_elem = ET.SubElement(entry, "Class")
                    class_elem.text = XMLDataProcessor._sanitize_xml_text(str(row.get("class", "Unknown")))
                    
                    field_intel_elem = ET.SubElement(entry, "Field_Intelligence_Translated")
                    field_intel_text = str(row.get("field_intelligence_translated", "No data"))
                    field_intel_elem.text = XMLDataProcessor._sanitize_xml_text(field_intel_text)
                    
                    date_elem = ET.SubElement(entry, "Submission Date")
                    date_elem.text = XMLDataProcessor._sanitize_xml_text(str(row.get("submission_datetime", "No date")))

                    # Add optional fields if they exist and are not null
                    location = row.get("location_name")
                    if pd.notna(location) and location != "N/A":
                        location_elem = ET.SubElement(entry, "Location")
                        location_elem.text = XMLDataProcessor._sanitize_xml_text(str(location))

                    # Add Form_Type and nested details
                    form_type_elem = ET.SubElement(entry, "Form_Type")
                    form_type_elem.text = XMLDataProcessor._sanitize_xml_text(str(row.get("form_type", "Unknown")))

                    form_type_details = ET.SubElement(entry, "Form_Type_Details")
                    for col in [
                        "product_category_name",
                        "brand_name_from",
                        "brand_name_to",
                        "pmi_product_name",
                        "tmo_name",
                    ]:
                        value = row.get(col)
                        if pd.notna(value) and value != "N/A":
                            detail_elem = ET.SubElement(form_type_details, col)
                            detail_elem.text = XMLDataProcessor._sanitize_xml_text(str(value))
                            
                except Exception as row_error:
                    logger.warning(f"Error processing row {idx}: {row_error}")
                    # Continue with next row instead of failing entirely
                    continue

            # Convert the tree to a string with error handling
            try:
                return ET.tostring(root, encoding="unicode", method="xml")
            except Exception as e:
                logger.error(f"Failed to convert XML tree to string: {e}")
                # Return a minimal XML structure as fallback
                return "<Entries><Entry><Row_ID>0</Row_ID><Class>Error</Class><Field_Intelligence_Translated>XML generation failed</Field_Intelligence_Translated></Entry></Entries>"
                
        except Exception as e:
            logger.error(f"Failed to create XML infos: {e}")
            # Return minimal valid XML
            return "<Entries><Entry><Row_ID>0</Row_ID><Class>Error</Class><Field_Intelligence_Translated>Data processing failed</Field_Intelligence_Translated></Entry></Entries>"
    
    @staticmethod
    def _sanitize_xml_text(text: str) -> str:
        """
        Sanitize text for XML to prevent parsing errors
        
        Parameters:
            text: Input text to sanitize
            
        Returns:
            str: Sanitized text safe for XML
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Handle None or empty strings
        if not text or text.strip() == "":
            return "N/A"
        
        # Remove or replace problematic characters
        # Replace control characters except tab, newline, and carriage return
        import unicodedata
        sanitized = ""
        for char in text:
            if unicodedata.category(char)[0] == "C" and char not in ['\t', '\n', '\r']:
                continue  # Skip control characters
            sanitized += char
        
        # Replace XML-problematic characters
        replacements = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&apos;'
        }
        
        for char, replacement in replacements.items():
            sanitized = sanitized.replace(char, replacement)
        
        # Limit length to prevent extremely long text
        if len(sanitized) > 5000:
            sanitized = sanitized[:4997] + "..."
        
        return sanitized


# Define a backoff handler to log retry attempts and update count