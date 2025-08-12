from dataclasses import asdict, dataclass
from typing import Any, List, Optional, Dict, Union
import streamlit as st
import math
import logging

logging.basicConfig(level=logging.WARNING)

# Initialize logger
logger = logging.getLogger(__name__)

# For backoff logging
logging.getLogger("backoff").addHandler(logging.StreamHandler())
logging.getLogger("backoff").setLevel(logging.WARNING)


@dataclass
class Message:
    content: str
    role: str
    avatar: Optional[str] = None
    documents: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_openai(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}

    def to_anthropic(self) -> Dict[str, str]:
        return {"role": self.role, "content": [{"text": self.content}]}


class ToolMessage(Message):
    def __init__(self, function_response: str, tool_call_id: str) -> None:
        super().__init__(content=function_response, role="tool")
        self.tool_call_id = tool_call_id

    def to_openai(self) -> dict[str, any]:
        return {
            "role": self.role,
            "content": str(self.content),
            "tool_call_id": self.tool_call_id,
        }


class ToolCallMessage(Message):
    def __init__(self, tool_call_id: str, tool_name: str, tool_args: dict) -> None:
        super().__init__(content=None, role="assistant")
        self.tool_calls = [
            {
                "id": tool_call_id,
                "type": "function",
                "function": {"name": tool_name, "arguments": tool_args},
            }
        ]

    def to_openai(self) -> dict[str, any]:
        return {"role": self.role, "tool_calls": self.tool_calls}


def stream_message_from_anthropic(
        executor, CONVERSATION_MODE=False
) -> tuple[str, Optional[float]]:
    message_placeholder = st.empty()
    full_response = ""
    logprobs = []

    # More aggressive error handling for the executor
    if executor is None:
        logger.error("Received None executor in stream_message_from_anthropic")
        return "Error: API response is not available", None

    # Check if executor is a callable before attempting to call it
    if not callable(executor):
        logger.error(f"Executor is not callable: {type(executor)}")
        return "Error: API response executor is not valid", None

    try:
        # Wrap the executor call in a try block
        executor_result = executor()

        # Ensure executor_result is iterable
        if executor_result is None:
            logger.error("Executor returned None instead of an iterable")
            return "Error: API returned an invalid response", None

        for response in executor_result:
            # Guard against None responses
            if response is None:
                logger.error("Received None response item from executor")
                continue

            # Make sure response is a dictionary before using .get()
            if not isinstance(response, dict):
                logger.error(f"Response is not a dictionary: {type(response)}")
                continue

            # Safely get event_type with a default value
            event_type = response.get("event_type", "")

            if event_type == "message_start":
                # Log the start of the message stream
                logger.debug("Message stream started.")
                continue

            elif event_type == "content_block_delta":
                # Safely get delta with a default empty dict
                delta = response.get("delta", {})

                # Make sure delta is a dictionary
                if not isinstance(delta, dict):
                    logger.error(f"Delta is not a dictionary: {type(delta)}")
                    continue

                # Check if content text is present
                if "text" in delta and delta["text"] is not None:
                    full_response += delta["text"]
                    # Use try-except to handle potential markdown rendering issues
                    try:
                        message_placeholder.markdown(full_response + "▌")
                    except Exception as markdown_error:
                        logger.error(f"Error rendering markdown: {str(markdown_error)}")
                    logger.debug("Received content block: %s", delta["text"])
    except Exception as e:
        logger.error(f"Error in stream_message_from_anthropic: {str(e)}")
        # Return partial response if we have any, otherwise return error message
        if not full_response:
            full_response = f"Error processing response: {str(e)}"

    # Calculate the probability if logprobs are available
    if logprobs:
        linear_prob = sum(
            [math.exp(lp) for lp in logprobs if isinstance(lp, (int, float))]
        ) / len(logprobs)
    else:
        linear_prob = None

    if not CONVERSATION_MODE:
        message_placeholder.markdown("")
    else:
        message_placeholder.markdown(full_response)
    return full_response, linear_prob


def is_function_call(response_or_func_call: Union[str, dict[str, str]]) -> bool:
    return isinstance(response_or_func_call, dict)
