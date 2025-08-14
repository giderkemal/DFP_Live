import pandas as pd
import streamlit as st
from dataclasses import dataclass
from typing import Any
from chat import Message


@dataclass
class StreamlitStateManager:
    client = st

    @classmethod
    def keys(self) -> list[str]:
        return list(self.client.session_state.keys())

    @classmethod
    def get(self, key: str, default: Any = None) -> Any:
        return self.client.session_state.get(key, default)

    @classmethod
    def set(self, key: str, value: Any) -> None:
        self.client.session_state[key] = value

    @classmethod
    def clear(self) -> None:
        self.client.session_state = {}


def add_message_to_state(message: Message) -> None:
    # Safely append to the messages list
    messages = StreamlitStateManager.get("messages", [])
    messages.append(message)
    StreamlitStateManager.set("messages", messages)


def get_messages_from_state(key: str = "messages") -> list[Message]:
    return StreamlitStateManager.get(key, [])


def clear_messages_from_state() -> None:
    StreamlitStateManager.set("messages", [])
    st.rerun()


def clear_filters() -> None:
    StreamlitStateManager.set("report_generated", False)
    StreamlitStateManager.set("report_displayed", False)
    StreamlitStateManager.set("messages", [])
    st.session_state.base_matrix_scope = pd.DataFrame()
    st.session_state.selected_filters = pd.DataFrame(
        columns=["Date range", "Feedback class", "Form type"]
    )
    st.rerun()


def set_bedrock_access_token() -> str:
    return StreamlitStateManager.get("access_token", "")


def initiate_state_variables():
    if "selected_filters" not in st.session_state:
        st.session_state.selected_filters = pd.DataFrame(
            columns=["Date range", "Feedback class", "Form type"]
        )

    if "base_matrix_scope" not in st.session_state:
        st.session_state.base_matrix_scope = pd.DataFrame()

    if "start_generation" not in st.session_state:
        st.session_state.start_generation = False