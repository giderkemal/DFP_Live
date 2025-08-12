import streamlit as st
import os
from dotenv import load_dotenv

import streamlit_utils.charts as charts
import streamlit_utils.frontend as frontend
import streamlit_utils.session as session

from models import ReportGeneration, ReportCombination

from base_matrix import BaseMatrixService
from retrieve_metadata import RetrieveMetadataService

load_dotenv()

# Force load from the specific .env file path
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'), override=True)

session.initiate_state_variables()

frontend.effixis_page_config()

unique_metadata_combinations = (
    RetrieveMetadataService.get_unique_metadata_combinations()
)

frontend.plain_gpt_intro()
filter_applied = frontend.set_filters(unique_metadata_combinations)

# Check if filters were applied
base_matrix = None
if not st.session_state.base_matrix_scope.empty:
    base_matrix, base_matrix_size = BaseMatrixService.create_base_matrix(
        st.session_state.base_matrix_scope
    )

    # Store filtered data for charts
    if 'base_matrix_scope' in st.session_state and not st.session_state.base_matrix_scope.empty:
        st.session_state['selected_filters'] = st.session_state.base_matrix_scope.copy()

    # Display visualizations at the top
    charts.display_charts(base_matrix)

    # Show limits message
    frontend.show_limits_message(base_matrix_size)

    # Set up intermediate reports if needed
    if session.StreamlitStateManager.get("generate_intermediate_reports"):
        ReportCombination.determine_intermediate_reports(
            base_matrix, st.session_state.base_matrix_scope
        )

    # Modified report generation section with minimized user notes
    if filter_applied:
        # Use a single spinner for the entire report generation process
        with st.spinner("Generating insights..."):
            if session.StreamlitStateManager.get("subgroups"):
                # Generate intermediate reports silently without individual success messages
                reports = []
                for subgroup in session.StreamlitStateManager.get("subgroups"):
                    reports.append(
                        ReportCombination.generate_intermediate_report(subgroup)
                    )
                # Combine reports without additional spinner
                ReportCombination.combine_reports(reports)
            else:
                # Generate single report directly
                ReportGeneration.generate_report(base_matrix)
            session.clear_messages_from_state()

# Display report if generated - WITHOUT IMAGES
if session.StreamlitStateManager.get(
        "report_generated"
) and session.StreamlitStateManager.get("generator_output"):

    output = session.StreamlitStateManager.get("generator_output")

    ReportGeneration.extract_infos_from_report(output)

    # Get the report content
    report = session.StreamlitStateManager.get("report")

    with st.container():
        html_content = """
        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px;'>
        {content}
        </div>
        """.format(
            content=report.replace("\n", "<br>")
        )
        st.markdown(html_content, unsafe_allow_html=True)
    session.StreamlitStateManager.set("report_displayed", True)

    if session.StreamlitStateManager.get("report_displayed"):
        frontend.display_followup_questions()

# Move the "Show source data" expander to the bottom of the page
if base_matrix is not None:
    with st.expander("Show source data"):
        st.write(base_matrix)