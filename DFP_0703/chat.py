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
    """
    Enhanced streaming function with comprehensive error handling and recovery.
    
    Parameters:
        executor: Function that returns an iterable of response chunks
        CONVERSATION_MODE: Whether to maintain the streaming display
    
    Returns:
        tuple: (full_response, probability) or error information
    """
    message_placeholder = st.empty()
    full_response = ""
    logprobs = []
    chunk_count = 0
    max_chunks = 10000  # Prevent infinite loops

    # More aggressive error handling for the executor
    if executor is None:
        logger.error("Received None executor in stream_message_from_anthropic")
        return "Error: API response is not available", None

    # Check if executor is a callable before attempting to call it
    if not callable(executor):
        logger.error(f"Executor is not callable: {type(executor)}")
        return "Error: API response executor is not valid", None

    try:
        # Wrap the executor call in a try block with timeout consideration
        try:
            executor_result = executor()
        except Exception as exec_error:
            logger.error(f"Executor call failed: {exec_error}")
            return f"Error: Failed to initiate streaming - {str(exec_error)}", None

        # Ensure executor_result is iterable
        if executor_result is None:
            logger.error("Executor returned None instead of an iterable")
            return "Error: API returned an invalid response", None

        # Check if executor_result is actually iterable
        try:
            iterator = iter(executor_result)
        except TypeError:
            logger.error(f"Executor result is not iterable: {type(executor_result)}")
            return "Error: API response is not in expected format", None

        # Process the streaming response with enhanced error handling
        for response in iterator:
            chunk_count += 1
            
            # Safety check to prevent infinite loops
            if chunk_count > max_chunks:
                logger.warning(f"Maximum chunk limit ({max_chunks}) reached, stopping stream")
                break
            
            # Guard against None responses
            if response is None:
                logger.warning(f"Received None response item at chunk {chunk_count}")
                continue

            # Make sure response is a dictionary before using .get()
            if not isinstance(response, dict):
                logger.warning(f"Response chunk {chunk_count} is not a dictionary: {type(response)}")
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
                    logger.warning(f"Delta in chunk {chunk_count} is not a dictionary: {type(delta)}")
                    continue

                # Check if content text is present
                if "text" in delta and delta["text"] is not None:
                    text_chunk = delta["text"]
                    
                    # Validate text chunk
                    if isinstance(text_chunk, str):
                        full_response += text_chunk
                        
                        # Use try-except to handle potential markdown rendering issues
                        try:
                            if not CONVERSATION_MODE:
                                message_placeholder.markdown(full_response + "▌")
                        except Exception as markdown_error:
                            logger.warning(f"Error rendering markdown at chunk {chunk_count}: {str(markdown_error)}")
                            # Try plain text as fallback
                            try:
                                message_placeholder.text(full_response + "▌")
                            except Exception as text_error:
                                logger.warning(f"Error rendering text fallback: {str(text_error)}")
                        
                        logger.debug(f"Processed chunk {chunk_count}: {len(text_chunk)} characters")
                    else:
                        logger.warning(f"Text chunk {chunk_count} is not a string: {type(text_chunk)}")
                        
            elif event_type == "message_stop":
                logger.debug("Message stream completed normally.")
                break
                
            elif event_type == "error":
                error_info = response.get("error", "Unknown error")
                logger.error(f"Stream error at chunk {chunk_count}: {error_info}")
                if not full_response:  # Only fail if we haven't received any content
                    return f"Error: Stream failed - {error_info}", None
                else:
                    logger.info("Partial response available despite stream error")
                    break
            
            else:
                # Log unexpected event types but continue processing
                logger.debug(f"Unexpected event type at chunk {chunk_count}: {event_type}")
                
    except StopIteration:
        logger.debug("Stream iteration completed.")
    except Exception as e:
        logger.error(f"Error in stream_message_from_anthropic: {str(e)}")
        # Return partial response if we have any, otherwise return error message
        if not full_response:
            return f"Error processing response: {str(e)}", None
        else:
            logger.info(f"Returning partial response due to error: {str(e)}")

    # Validate the final response
    if not full_response or len(full_response.strip()) < 10:
        logger.warning("Response is empty or too short")
        if full_response:
            return full_response, None  # Return what we have
        else:
            return "Error: No content received from API", None

    # Calculate the probability if logprobs are available
    if logprobs:
        try:
            linear_prob = sum(
                [math.exp(lp) for lp in logprobs if isinstance(lp, (int, float))]
            ) / len(logprobs)
        except (ValueError, ZeroDivisionError) as e:
            logger.warning(f"Error calculating probability: {e}")
            linear_prob = None
    else:
        linear_prob = None

    # Clear or update the final display
    try:
        if not CONVERSATION_MODE:
            message_placeholder.markdown("")
        else:
            message_placeholder.markdown(full_response)
    except Exception as display_error:
        logger.warning(f"Error updating final display: {display_error}")

    logger.info(f"Stream completed successfully: {len(full_response)} characters, {chunk_count} chunks processed")
    return full_response, linear_prob


def is_function_call(response_or_func_call: Union[str, dict[str, str]]) -> bool:
    return isinstance(response_or_func_call, dict)
