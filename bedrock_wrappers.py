from typing import List, Dict, Generator
import requests
import logging
import backoff
import time

# Initialize logger
logger = logging.getLogger(__name__)

# For backoff logging
logging.getLogger("backoff").addHandler(logging.StreamHandler())
logging.getLogger("backoff").setLevel(logging.WARNING)

# Track retry count with a mutable object
retry_tracker = {"count": 0}


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


PRIVATE_API = False


class APIManager:
    """
    Manages API requests by dynamically selecting between public and private endpoints.
    """

    PUBLIC_BASE_URL = "https://aiplat.aws.pmicloud.biz"
    PRIVATE_BASE_URL = "https://aiplat.aws.private-pmideep.biz"

    def __init__(self, private: bool = False):
        self.base_url = self.PRIVATE_BASE_URL if private else self.PUBLIC_BASE_URL

    def api_endpoint(self, path: str) -> str:
        return f"{self.base_url}/api/{path}"


def api_chat_completions(
    access_token: str,
    messages: List[Dict],
    model: str,
    max_tokens: int = 1000,
    temperature: float = 0,
    top_p: float = 1,
    private_api: bool = PRIVATE_API,
):
    api_manager = APIManager(private=private_api)

    # FORM REQUEST HEADERS AND PAYLOAD
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": 0.7, "topP": 1},
        "messages": messages,
        "modelId": model,
    }

    # CALL API
    return requests.post(
        url=api_manager.api_endpoint("bedrock-runtime/v1/converse"),
        headers=headers,
        json=payload,
    )


def api_chat_completions_stream(
    access_token: str,
    messages: List[Dict],
    model: str,
    max_tokens: int = 1000,
    temperature: float = 0,
    top_p: float = 1,
    private_api: bool = PRIVATE_API,
    verbose: bool = False,
) -> Generator[Dict, None, None]:
    """
    Stream chat completions using a previously obtained access token.

    :param access_token: Access token for authorization.
    :param messages: List of message dictionaries for the model to process.
    :param model: Model ID for chat completion.
    :param max_tokens: Maximum tokens for generation.
    :param temperature: Sampling temperature.
    :param top_p: Top-p sampling parameter.
    :param verbose: If True, enables detailed logging.
    :yield: JSON response chunks from the streaming endpoint.
    """
    # Set logging level based on verbose parameter
    logging.basicConfig(level=logging.WARNING)

    # Form request headers and payload
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "inferenceConfig": {
            "maxTokens": max_tokens,
            "temperature": temperature,
            "topP": top_p,
        },
        "messages": messages,
        "modelId": model,
    }

    if private_api:
        url = "https://aiplat.aws.private-pmideep.biz/api/bedrock-runtime/v1/converse_stream"
    else:
        url = "https://aiplat.aws.pmicloud.biz/api/bedrock-runtime/v1/converse_stream"
    # Start stream and retrieve stream_id
    start_response = requests.post(
        f"{url}/start",
        headers=headers,
        json=payload,
    )
    start_response.raise_for_status()
    stream_id = start_response.json().get("ResponseMetadata", {}).get("RequestId")

    if not stream_id:
        logger.error("Stream initiation failed: Stream ID not retrieved.")
        raise ValueError("Stream initiation failed: Stream ID not retrieved.")

    routing_cookie = start_response.cookies.get("routing")

    if not routing_cookie:
        logger.error("Stream initiation failed: Routing cookie not retrieved.")
        raise ValueError("Stream initiation failed: Routing cookie not retrieved.")

    cookies = {"routing": routing_cookie}

    logger.info("Stream initiated successfully. Stream ID: %s", stream_id)
    logger.debug("Routing cookie: %s", routing_cookie)

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
        max_time=60,
        max_tries=5,
        giveup=fatal_code,
        raise_on_giveup=True,
        on_backoff=backoff_hdlr,
    )
    def response_generator() -> Generator[Dict, None, None]:
        next_url = f"{url}/next/{stream_id}"
        logger.info("Backoff: Current retry attempt: %d", retry_tracker["count"])
        while True:
            # Get next chunk of data
            start_time = time.time()
            next_response = requests.get(next_url, headers=headers, cookies=cookies)
            elapsed_time = time.time() - start_time
            server_time = (
                next_response.json().get("api_metrics").get("total_response_time")
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
