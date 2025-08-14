import pandas as pd
import streamlit as st
import streamlit_utils.session as session
from models import ConnectionAPI
from chat import Message, ToolCallMessage, stream_message_from_anthropic


def effixis_page_config() -> None:
    st.set_page_config(layout="wide")  # Sets the layout to wide mode

    with st.sidebar:
        # check if authenticated
        st.image(image="assets/pmi_logo.png")


def clear_conversation_button() -> bool:
    return st.button(
        "**Restart Conversation** 🔄",
        on_click=session.clear_messages_from_state,
        key="clear_conversation_btn",
    )


def plain_gpt_intro():
    # Customized header with CSS for a patent search tool
    st.markdown(
        """
        <style>
            .header {
                font-size: 28px;
                color: #003f5c; # TODO change color
                text-shadow: 1px 1px 3px #333;
            }
        </style>
        <div class="header">
            Field Intelligence Report Generation Tool 📊
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """Welcome to the Field Intelligence Report Generation Tool, your go-to solution for categorizing and analyzing field data with ease. This tool leverages cutting-edge AI to process and classify insights from diverse inputs, helping you turn raw information into actionable reports. Whether you're streamlining workflows, enhancing decision-making, or exploring patterns in field intelligence, this tool is designed to simplify and elevate your data analysis experience."""
    )
    st.divider()


def display_message(message: Message) -> None:
    if message.role == "system":
        if message.documents is not None and len(message.documents):
            with st.expander("**📄 Show Retrieved Documents**"):
                for i, document in enumerate(
                        sorted(message.documents, key=lambda x: x["page_no"])
                ):
                    if 10 > 0 and i >= 10:
                        break
                    st.subheader(document["title"])
                    st.markdown(f"**Passage:** {document['passage']}")
                    st.markdown("---")
        else:
            if len(message.content) > 0:
                with st.expander("Function Response 📦"):
                    st.markdown(
                        message.content
                    )  # Display system messages with an expander

    elif message.role == "assistant":
        if isinstance(message, ToolCallMessage) is False:
            if message.content is not None:
                with st.chat_message(message.role):
                    st.write(message.content)
        else:
            # TODO : persistant tool call display
            pass
    elif message.role == "user":
        if not message.content.startswith(
                "Please remember the information in the following documents:"
        ):
            with st.chat_message(message.role):
                st.write(message.content)

    elif message.role == "tool":
        # TODO : persistant tool response display
        pass


def display_messages() -> None:
    messages = session.get_messages_from_state()
    for message in messages:
        if message.role != "system":
            display_message(message)


def get_user_input():
    user_input = st.chat_input("Type your message here ...")
    if user_input:
        user_message = Message(content=user_input, role="user", avatar="user")
        session.add_message_to_state(user_message)
        return user_input
    return None


# Modified set_filters function in frontend.py

def set_filters(metadata):
    date_col, feedback_class_col, form_type_col = st.columns([1, 1, 1])

    with date_col:
        select_date_range(
            label="Select date range",
            key="date_range",
            start=min(metadata["DATE"]),
            end=max(metadata["DATE"]),
        )

    with feedback_class_col:
        st.multiselect(
            label="Feedback class",
            key="feedback_class",
            options=metadata["CLASS"].dropna().unique().tolist(),
        )

    with form_type_col:
        form_type = st.selectbox(
            label="Form type",
            key="form_type",
            options=["All"] + metadata["FORM_TYPE"].dropna().unique().tolist(),
        )

    form_type_filters = {
        "All": ["Region", "Market", "Location", "TMO", "Brand"],
        "CROSS_CATEGORY": ["Region", "Market", "Location", "Product Category"],
        "CONSUMER_FEEDBACK": [
            "Region",
            "Market",
            "Location",
            "Answer Category",
            "PMI Product",
        ],
        "TOBACCO_CATEGORY": [
            "Region",
            "Market",
            "Location",
            "TMO",
            "Brand",
            "Product Category",
        ],
        "BRAND_SOURCING": ["Region", "Market", "Location", "Switch From", "Switch To"],
        "INFRA_MAINTENANCE": ["Region", "Market", "Location"],
    }

    filters_list = form_type_filters[form_type]
    filter_columns = st.columns(len(filters_list))

    filtered_metadata = metadata.copy()
    dynamic_filters = {}

    for col, filter_name in zip(filter_columns, filters_list):
        with col:
            filter_key = f"filter_{filter_name}"
            current_selection = st.session_state.get(filter_key, [])

            options = sorted(filtered_metadata[filter_name].dropna().unique().tolist())

            safe_default = [val for val in current_selection if val in options]
            if len(current_selection) != len(safe_default):
                st.session_state[filter_key] = safe_default

            selected = st.multiselect(
                label=f"{filter_name}",
                options=options,
                default=safe_default,
                key=filter_key
            )

            dynamic_filters[filter_name] = selected

            if selected:
                filtered_metadata = filtered_metadata[
                    filtered_metadata[filter_name].isin(selected)
                ]

    # Return True if Apply button was clicked, False otherwise
    filter_applied = st.button("Apply", use_container_width=True)

    if filter_applied:
        scope = add_filter(dynamic_filters)
        st.session_state.base_matrix_scope = scope
        # Also update selected_filters for backward compatibility
        st.session_state.selected_filters = scope

    return filter_applied

def add_filter(dynamic_filters):
    current_filter_df = pd.DataFrame()

    current_filter_df.at[0, "Date range"] = " \u2014 ".join(
        map(str, (st.session_state.date_range[0], st.session_state.date_range[1]))
    )
    current_filter_df.at[0, "Feedback class"] = (
        ", ".join(st.session_state.feedback_class)
        if st.session_state.feedback_class
        else "All"
    )

    current_filter_df.at[0, "Form type"] = str(st.session_state.form_type)

    for k, v in dynamic_filters.items():
        current_filter_df.at[0, k] = ", ".join(v) if v else "All"

    # Update both session state variables to ensure compatibility
    selected_filters = pd.concat(
        [pd.DataFrame(), current_filter_df],  # Start with a fresh DataFrame
        ignore_index=True,
    )

    return selected_filters


def select_date_range(label, key, start, end, **kwargs):
    dates = st.date_input(label, key=key, value=(start, end), **kwargs)

    if isinstance(dates, (list, tuple)) and len(dates) == 2:
        return dates

    return start, end


def show_limits_message(base_matrix_size, limit=1200):
    if base_matrix_size > limit:
        session.StreamlitStateManager.set("start_generation", False)
        session.StreamlitStateManager.set("generate_intermediate_reports", True)
        st.warning(
            f"Dataset exceeds limit. Combined generation used."
        )
    elif base_matrix_size == 0:
        session.StreamlitStateManager.set("start_generation", False)
        st.error(f"Empty dataset. Adjust scope.")
    else:
        session.StreamlitStateManager.set("start_generation", True)
        # Completely removed success message for when scope is applied


def display_followup_questions():
    """
    Main function to handle the display and processing of follow-up questions in the Streamlit session.
    """

    if not session.StreamlitStateManager.get("report_displayed"):
        return

    st.markdown("### Ask follow-up questions:")

    # Retrieve session data
    data, report, user_input = get_session_data()
    if not user_input:
        return

    # Ensure access token is available
    ensure_access_token()

    clear_conversation_button()

    # Construct system prompt
    system_prompt = construct_system_prompt(data, report)

    process_followup_questions(system_prompt)


def get_session_data():
    """Retrieve required session data."""
    data = session.StreamlitStateManager.get("data")
    report = session.StreamlitStateManager.get("report")
    user_input = get_user_input()
    return data, report, user_input


def ensure_access_token():
    """Ensure the session has a valid access token."""
    access_token = session.StreamlitStateManager.get("access_token")
    if not access_token:
        access_token = ConnectionAPI.get_access_token()
        session.StreamlitStateManager.set("access_token", access_token)


def construct_system_prompt(data, report):
    """Constructs the system prompt for the assistant."""
    return f"""
    You are a data analyst specializing in employee feedback and customer request management.
    The following report was generated:
    '''
    {report}
    '''
    The data used to generate this report is:
    '''
    {data}
    '''

    The user will ask you follow-up questions. Please answer them to the best of your ability.
    Let's think step by step.
    """


def process_followup_questions(system_prompt):
    """Handles the interaction with the assistant for follow-up questions."""
    max_attempts = 3

    for attempt in range(1, max_attempts + 1):
        try:
            messages = session.get_messages_from_state()

            # Remove last assistant message if retrying
            if attempt > 1 and messages and messages[-1].role == "assistant":
                messages.pop()

            # Token estimation check
            estimated_tokens = ConnectionAPI.count_tokens_with_anthropic(
                messages, system_prompt
            )
            if estimated_tokens > 150_000:
                st.error(
                    "Context too long. Restart conversation."
                )
                return

            # API call for chat completion
            executor = ConnectionAPI.api_chat_completions_stream_message(
                messages=[message.to_anthropic() for message in messages],
                system_prompt=system_prompt,
            )

            # Check if executor is valid
            if executor is None or not callable(executor):
                # Try again if we have more attempts left
                if attempt < max_attempts:
                    continue
                st.error("Connection failed. Try reloading.")
                return

            # Display response
            display_messages()
            with st.chat_message("assistant"):
                response_or_func_call, prob = stream_message_from_anthropic(
                    executor, CONVERSATION_MODE=True
                )
                # Check if response contains error message
                if response_or_func_call and not response_or_func_call.startswith("Error:"):
                    session.add_message_to_state(
                        Message(
                            content=response_or_func_call, role="assistant", avatar="🤖"
                        )
                    )
                else:
                    # Retry silently if we hit an error and have attempts left
                    if attempt < max_attempts:
                        continue
                    # On last attempt, show a simple error
                    st.error("Connection failed. Try reloading.")
                    return
            return

        except Exception as e:
            # Last attempt, show error
            if attempt == max_attempts:
                st.error(
                    "Connection failed. Try reloading."
                )
                st.stop()