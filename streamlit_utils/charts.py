import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re
import numpy as np

# Import necessary service to get the unfiltered data
from snowflake_connection import SnowflakeConnectionService


def display_charts(filtered_matrix):
    """
    Displays enhanced charts based on the dataset:
    1. Geographical hierarchy sunburst chart (Region > Market > Location) - showing ALL data but highlighting selected
    2. Time series chart of form submissions - using FILTERED data

    Args:
        filtered_matrix (pd.DataFrame): The filtered dataset (already filtered by date and other criteria)
    """
    if filtered_matrix is None or filtered_matrix.empty:
        return

    # Create two columns for side-by-side charts
    col1, col2 = st.columns(2)

    # 1. Geographical Hierarchy Sunburst Chart - showing ALL data but highlighting selected
    with col1:
        try:
            # Get the FULL unfiltered dataset to show everything in sunburst
            full_matrix = SnowflakeConnectionService.get_raw_base_matrix()

            if full_matrix is None or full_matrix.empty:
                # Fallback to filtered matrix if we can't get full data
                full_matrix = filtered_matrix.copy()

            # Apply only date and form type filters (NOT geographical filters)
            filtered_scope = st.session_state.get('selected_filters', pd.DataFrame())
            if filtered_scope.empty:
                filtered_scope = st.session_state.get('base_matrix_scope', pd.DataFrame())

            # Start with full data for sunburst display
            sunburst_data = full_matrix.copy()

            # Extract selected filters for dynamic title creation
            selected_region = None
            selected_market = None
            selected_location = None
            selected_form_types = None
            date_range_text = ""

            # Check first if there are any filters applied
            if not filtered_scope.empty:
                # Look for selected form type
                if 'Form type' in filtered_scope.columns:
                    form_type_value = filtered_scope['Form type'].iloc[0]
                    if isinstance(form_type_value, str) and form_type_value != 'All':
                        selected_form_types = [x.strip() for x in form_type_value.split(',')]

                # Look for selected region (for HIGHLIGHTING only)
                if 'Region' in filtered_scope.columns:
                    region_value = filtered_scope['Region'].iloc[0]
                    if isinstance(region_value, str) and region_value != 'All':
                        selected_region = [x.strip() for x in region_value.split(',')]

                # Look for selected market (for HIGHLIGHTING only)
                if 'Market' in filtered_scope.columns:
                    market_value = filtered_scope['Market'].iloc[0]
                    if isinstance(market_value, str) and market_value != 'All':
                        selected_market = [x.strip() for x in market_value.split(',')]

                # Look for selected location (for HIGHLIGHTING only)
                if 'Location' in filtered_scope.columns:
                    location_value = filtered_scope['Location'].iloc[0]
                    if isinstance(location_value, str) and location_value != 'All':
                        selected_location = [x.strip() for x in location_value.split(',')]

                # Extract date range for title
                if 'Date range' in filtered_scope.columns:
                    date_range = filtered_scope['Date range'].iloc[0]
                    if isinstance(date_range, str) and '—' in date_range:
                        start_date, end_date = date_range.split('—')
                        start_date = pd.to_datetime(start_date.strip())
                        end_date = pd.to_datetime(end_date.strip())

                        sunburst_data['SUBMISSION_DATETIME'] = pd.to_datetime(
                            sunburst_data['SUBMISSION_DATETIME'])
                        sunburst_data = sunburst_data[
                            (sunburst_data['SUBMISSION_DATETIME'] >= start_date) &
                            (sunburst_data['SUBMISSION_DATETIME'] <= end_date)
                            ]

                # Apply form type filter ONLY (since we want to show distribution by geography for selected form type)
                if selected_form_types:
                    sunburst_data = sunburst_data[sunburst_data['FORM_TYPE'].isin(selected_form_types)]

            # Define region colors (fixed palette)
            region_colors = {
                'Japan and Korea': '#FF66B2',  # Pink
                'Americas': '#9966CC',  # Purple
                'Europe': '#3366CC',  # Dark Blue
                'China and SEA': '#CC3333',  # Red
                'Eurasia': '#FFCC33',  # Yellow
                'Middle East': '#33CC66',  # Green
                'Africa': '#FF9933'  # Orange
            }

            # Default color for regions not in our defined list
            default_color = '#CCCCCC'  # Gray
            dimmed_color = '#E6E6E6'  # Dimmed color for non-selected regions (lighter gray)

            # Clean and prepare data - replace NaN values
            sunburst_data['VP_REGION_NAME'] = sunburst_data['VP_REGION_NAME'].fillna('Not Specified')
            sunburst_data['DF_MARKET_NAME'] = sunburst_data['DF_MARKET_NAME'].fillna('Not Specified')
            sunburst_data['LOCATION_NAME'] = sunburst_data['LOCATION_NAME'].fillna('Not Specified')

            # Group by geo hierarchy - SHOW ALL DATA, not just selected
            geo_counts = sunburst_data.groupby(
                ['VP_REGION_NAME', 'DF_MARKET_NAME', 'LOCATION_NAME']).size().reset_index()
            geo_counts.columns = ['Region', 'Market', 'Location', 'Count']

            if not geo_counts.empty:
                # Define a function to determine color based on selection (for highlighting)
                def get_highlight_color(row):
                    region = row['Region']
                    market = row['Market']
                    location = row['Location']

                    # Get the base color for the region
                    base_color = region_colors.get(region, default_color)

                    # If no geographical filters are active, use base colors for all
                    if not selected_region and not selected_market and not selected_location:
                        return base_color

                    # Determine if this row should be highlighted (bright) vs dimmed
                    should_highlight = True

                    # Check region filter - if a region is selected, only highlight that region
                    if selected_region and region not in selected_region:
                        should_highlight = False

                    # Check market filter - if a market is selected, only highlight that market
                    if should_highlight and selected_market and market not in selected_market:
                        should_highlight = False

                    # Check location filter - if a location is selected, only highlight that location
                    if should_highlight and selected_location and location not in selected_location:
                        should_highlight = False

                    # Return bright color for selected items, dimmed for non-selected
                    return base_color if should_highlight else dimmed_color

                # Apply the color function to each row
                geo_counts['Color'] = geo_counts.apply(get_highlight_color, axis=1)

                # Create a discrete color map that includes both bright and dimmed colors
                color_map = {}
                for color in geo_counts['Color'].unique():
                    color_map[color] = color

                # Create dynamic chart title
                chart_title = "Form Type Distribution by Geography"

                # Add geography selection to title
                geography_parts = []
                if selected_location:
                    geography_parts.extend(selected_location)
                elif selected_market:
                    geography_parts.extend(selected_market)
                elif selected_region:
                    geography_parts.extend(selected_region)

                # Add form type to title (remove underscores)
                form_type_parts = []
                if selected_form_types:
                    form_type_parts = [ft.replace('_', ' ') for ft in selected_form_types]

                # Build title with geography and form type
                if geography_parts and form_type_parts:
                    chart_title += f": {', '.join(geography_parts)} - {', '.join(form_type_parts)}"
                elif geography_parts:
                    chart_title += f": {', '.join(geography_parts)}"
                elif form_type_parts:
                    chart_title += f": {', '.join(form_type_parts)}"

                st.markdown(f"#### {chart_title}")

                # Create sunburst chart (NO subtitle)
                fig = px.sunburst(
                    geo_counts,
                    path=['Region', 'Market', 'Location'],
                    values='Count',
                    color='Color',
                    color_discrete_map=color_map,
                    hover_data=['Count']
                )

                # Custom hover template
                fig.update_traces(
                    hovertemplate='<b>%{id}</b><br>Count: %{value}<extra></extra>',
                    marker=dict(
                        line=dict(width=2, color='white'),  # Bold white borders
                    )
                )

                # Customize chart appearance (NO title in plotly chart)
                fig.update_layout(
                    margin=dict(t=10, l=0, r=0, b=0),  # Minimal margin
                    height=500,
                    uniformtext=dict(minsize=10, mode='hide')
                )

                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error creating sunburst chart: {e}")

    # 2. Time Series Chart - using FILTERED data
    with col2:
        try:
            # Get filter information for dynamic title
            filtered_scope = st.session_state.get('selected_filters', pd.DataFrame())
            if filtered_scope.empty:
                filtered_scope = st.session_state.get('base_matrix_scope', pd.DataFrame())

            # Create dynamic title for time series
            time_chart_title = "Form Submissions Over Time"

            title_parts = []

            # Add form type to title (remove underscores)
            if not filtered_scope.empty and 'Form type' in filtered_scope.columns:
                form_type_value = filtered_scope['Form type'].iloc[0]
                if isinstance(form_type_value, str) and form_type_value != 'All':
                    form_types = [ft.replace('_', ' ') for ft in form_type_value.split(',')]
                    title_parts.append(', '.join(form_types))

            # Add date range to title
            if not filtered_scope.empty and 'Date range' in filtered_scope.columns:
                date_range = filtered_scope['Date range'].iloc[0]
                if isinstance(date_range, str) and '—' in date_range:
                    start_date, end_date = date_range.split('—')
                    start_date = pd.to_datetime(start_date.strip()).strftime('%b %d, %Y')
                    end_date = pd.to_datetime(end_date.strip()).strftime('%b %d, %Y')
                    title_parts.append(f"{start_date} - {end_date}")

            # Build final title
            if title_parts:
                time_chart_title += f": {' - '.join(title_parts)}"

            st.markdown(f"#### {time_chart_title}")

            if 'SUBMISSION_DATETIME' in filtered_matrix.columns:
                # Prepare data for time series
                time_df = filtered_matrix.copy()

                # Try to convert SUBMISSION_DATETIME to datetime if it's not already
                time_df['SUBMISSION_DATETIME'] = pd.to_datetime(time_df['SUBMISSION_DATETIME'])
                # Extract just the date part
                time_df['SUBMISSION_DATE'] = time_df['SUBMISSION_DATETIME'].dt.date

                def ensure_min_points(dates, min_points=4):
                    """Ensure specific percentage-based points are shown on the time series x-axis."""
                    dates = pd.to_datetime(dates)

                    # Always create a date range from min to max date
                    all_dates = pd.date_range(start=min(dates), end=max(dates), freq='D')

                    # Always ensure we have at least min_points
                    if len(all_dates) < min_points:
                        # If we have fewer dates than minimum points,
                        # extend the range to ensure we have at least min_points
                        days_to_add = min_points - len(all_dates)
                        extended_end = max(dates) + pd.Timedelta(days=days_to_add)
                        all_dates = pd.date_range(start=min(dates), end=extended_end, freq='D')

                    # Use the min and max dates, plus points at 20%, 40%, 60%, and 80% positions
                    percentages = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

                    # Calculate indices based on percentages
                    indices = [int(p * (len(all_dates) - 1)) for p in percentages]

                    # Select the dates at these specific percentile positions
                    return all_dates[indices].tolist()

                # Group by date (single line, not by form type)
                time_series = time_df.groupby(['SUBMISSION_DATE']).size().reset_index()
                time_series.columns = ['Date', 'Count']

                if not time_series.empty:
                    # Get month-year combinations
                    time_series['Month_Year'] = pd.to_datetime(time_series['Date']).dt.strftime('%Y-%m')

                    # Get one date per month to show larger markers
                    monthly_dates = time_series.groupby('Month_Year')['Date'].first().tolist()

                    # Use all available dates with specific percentage-based points (0%, 20%, 40%, 60%, 80%, 100%)
                    highlighted_dates = ensure_min_points(time_series['Date'].tolist(), min_points=6)

                    # Create a smoother time series using a custom Graph Objects approach
                    fig = go.Figure()

                    # Add smoother line
                    fig.add_trace(
                        go.Scatter(
                            x=time_series['Date'],
                            y=time_series['Count'],
                            mode='lines',
                            line=dict(
                                color='#3366CC',
                                width=2,
                                shape='spline',  # Use spline for smoother curves
                                smoothing=0.3  # Adjust smoothing factor (0-1.3)
                            ),
                            name='Submissions'
                        )
                    )

                    # Add highlighted markers (larger, more visible)
                    fig.add_trace(
                        go.Scatter(
                            x=[date for date in time_series['Date'] if date in highlighted_dates],
                            y=[time_series.loc[time_series['Date'] == date, 'Count'].iloc[0] for date in
                               time_series['Date']
                               if date in highlighted_dates],
                            mode='markers',
                            marker=dict(
                                color='#3366CC',
                                size=8,
                                line=dict(width=1, color='white')
                            ),
                            showlegend=False,
                            hoverinfo='skip'
                        )
                    )

                    # Add smaller markers for all other points
                    fig.add_trace(
                        go.Scatter(
                            x=[date for date in time_series['Date'] if date not in highlighted_dates],
                            y=[time_series.loc[time_series['Date'] == date, 'Count'].iloc[0] for date in
                               time_series['Date']
                               if date not in highlighted_dates],
                            mode='markers',
                            marker=dict(
                                color='#3366CC',
                                size=4,
                                opacity=0.6,
                                line=dict(width=0.5, color='white')
                            ),
                            showlegend=False,
                            hoverinfo='skip'
                        )
                    )

                    # Enhance the appearance
                    fig.update_layout(
                        xaxis_title='Submission Date',
                        yaxis_title='Number of Submissions',
                        hovermode='x unified',
                        plot_bgcolor='rgba(255,255,255,1)',
                        height=500,
                        margin=dict(t=5, l=5, r=5, b=5),
                        showlegend=False,
                    )

                    # Update the x-axis configuration to show points at 0%, 20%, 40%, 60%, 80%, 100% positions
                    fig.update_xaxes(
                        tickangle=-45,
                        tickformat='%b %d',  # Month day format
                        type='date',
                        tickmode='array',
                        tickvals=highlighted_dates,  # Use our percentage-based points
                        gridcolor='lightgray',
                        showgrid=True,
                    )

                    # Format y-axis
                    fig.update_yaxes(
                        gridcolor='lightgray',
                        showgrid=True,
                    )

                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Time series data not available in the current selection.")
        except Exception as e:
            st.error(f"Error creating time series chart: {e}")