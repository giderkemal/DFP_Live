import streamlit as st
import pandas as pd
from typing import List, Union, Optional, Dict, Any
import logging
import re
import os
from s3_service import get_signed_url

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_row_ids_from_text(text: str) -> List[Union[int, str]]:
    """
    Extract row IDs from text formatted as [Row_ID:123]
    """
    if not text:
        return []

    row_ids = []
    # Look for [Row_ID:123] pattern
    pattern = r'\[Row_ID:(\d+)\]'
    matches = re.findall(pattern, text)

    # Convert to integers
    for match in matches:
        try:
            row_ids.append(int(match))
        except ValueError:
            # If conversion fails, keep as string
            row_ids.append(match)

    logger.info(f"Extracted {len(row_ids)} row IDs from text")
    return row_ids


def debug_s3_path(row: pd.Series, s3_images_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Helper function to debug S3 path construction.

    Args:
        row: A row from the main DataFrame
        s3_images_df: DataFrame with S3 image metadata

    Returns:
        Dictionary with debug information
    """
    debug_info = {
        'has_photo_link': False,
        'has_date': False,
        'has_account_code': False,
        'has_template_code': False,
        'constructed_path': None,
        'matching_s3_keys': [],
        'error': None
    }

    try:
        # Check PHOTO_LINK
        if pd.notna(row.get('PHOTO_LINK')) and row['PHOTO_LINK']:
            debug_info['has_photo_link'] = True

        # Check submission date
        if pd.notna(row.get('SUBMISSION_DATETIME')):
            debug_info['has_date'] = True
            submission_date = pd.to_datetime(row['SUBMISSION_DATETIME'])
            debug_info['year'] = str(submission_date.year)
            debug_info['month'] = str(submission_date.month)

        # Check account_code
        if pd.notna(row.get('ACCOUNT_CODE')):
            debug_info['has_account_code'] = True
            debug_info['account_code'] = row['ACCOUNT_CODE']

        # Check template_code
        if pd.notna(row.get('TEMPLATE_CODE')):
            debug_info['has_template_code'] = True
            debug_info['template_code'] = row['TEMPLATE_CODE']

        # Construct path pattern
        if debug_info['has_date'] and debug_info['has_account_code'] and debug_info['has_template_code']:
            path_pattern = f"{debug_info['year']}/{debug_info['month']}/{debug_info['account_code']}/{debug_info['template_code']}/"
            debug_info['constructed_path'] = path_pattern

            # Find matches in S3 image data
            matches = []
            for _, s3_row in s3_images_df.iterrows():
                s3_key = s3_row.get('S3_KEY', '')
                if s3_key and path_pattern in s3_key:
                    matches.append(s3_key)

            debug_info['matching_s3_keys'] = matches
            debug_info['match_count'] = len(matches)

    except Exception as e:
        debug_info['error'] = str(e)

    return debug_info


def find_matching_images(row: pd.Series) -> List[str]:
    """
    Find matching images for a row using S3 key data with Vault credentials.

    Args:
        row: A row from the main DataFrame with joined image data

    Returns:
        List of S3 image URLs
    """
    from snowflake_connection import SnowflakeConnectionService

    matches = []
    bucket_name = f'vapafpdf-{os.getenv("FDF_ENV", "dev")}-ds-storage'

    # Check for directly joined S3 key
    if pd.notna(row.get('matched_s3_key')):
        logger.info(f"Found direct S3 key match: {row['matched_s3_key']}")

        # Generate URL for this key using Vault-based S3 access
        image_url = get_signed_url(row['matched_s3_key'], bucket_name)
        if image_url:
            matches.append(image_url)
            logger.info(f"Successfully generated image URL from matched S3 key")
            return matches

    # If we don't have a matched S3 key but have a photo indicator,
    # try to find images directly from ISA_S3_IMAGES
    has_photo = row.get('HAS_IMAGE', False) or (pd.notna(row.get('PHOTO_LINK')) and row['PHOTO_LINK'])

    if has_photo:
        try:
            # Extract key fields for manual lookup
            year = int(row.get('YEAR', 0))
            month = int(row.get('MONTH_NUM', 0))
            template_code = row.get('TEMPLATE_CODE')
            account_code = row.get('ACCOUNT_CODE')
            task_code = row.get('TASK_CODE')

            # If we have the necessary fields, query directly
            if year > 0 and month > 0 and (template_code or task_code):
                # Build query conditions
                conditions = []
                if year:
                    conditions.append(f"YEAR = {year}")
                if month:
                    conditions.append(f"MONTH_NUM = {month}")
                if template_code:
                    conditions.append(f"TEMPLATE_CODE = '{template_code}'")
                if account_code:
                    conditions.append(f"ACCOUNT_CODE = '{account_code}'")
                if task_code:
                    conditions.append(f"TASK_ATTACHMENT_CODE = '{task_code}'")

                where_clause = " AND ".join(conditions)

                query = f"""
                SELECT S3_KEY 
                FROM INFORMATION_MART.ISA_S3_IMAGES
                WHERE {where_clause}
                LIMIT 3
                """

                img_df = SnowflakeConnectionService.fetch_query_result(query)

                if not img_df.empty:
                    # Process the results
                    for _, img_row in img_df.iterrows():
                        s3_key = img_row.get('S3_KEY')
                        if s3_key:
                            image_url = get_signed_url(s3_key, bucket_name)
                            if image_url:
                                matches.append(image_url)

                    if matches:
                        return matches
        except Exception as e:
            logger.error(f"Error in manual lookup: {e}")

    return matches


def display_images_for_report(df: pd.DataFrame, row_ids: List[Union[int, str]], max_total: int = 6):
    """
    Displays images for rows referenced in the report using directly joined image data.

    Args:
        df: DataFrame containing required columns with joined image data
        row_ids: List of row IDs mentioned in the report
        max_total: Maximum total number of images to display
    """
    logger.info(f"Displaying images for {len(row_ids) if row_ids else 0} row IDs")

    if df is None or df.empty:
        logger.warning("No data available to retrieve images")
        return

    if not row_ids:
        logger.info("No row IDs provided for image display")
        return

    # Clean row_ids
    cleaned_row_ids = []
    for row_id in row_ids:
        if row_id is None:
            continue
        try:
            if isinstance(row_id, str):
                row_id = row_id.strip()
                if row_id.isdigit():
                    cleaned_row_ids.append(int(row_id))
                else:
                    cleaned_row_ids.append(row_id)
            else:
                cleaned_row_ids.append(row_id)
        except Exception as e:
            logger.error(f"Error processing row ID {row_id}: {str(e)}")
            continue

    if not cleaned_row_ids:
        logger.warning("No valid row IDs found after cleaning")
        return

    # Check for image data columns
    has_image_data = 'matched_s3_key' in df.columns or 'HAS_IMAGE' in df.columns
    if not has_image_data:
        logger.warning("DataFrame does not contain joined image data columns")
        st.warning("Image data not available in the current view. Try refreshing the page.")
        return

    # Add some debugging info
    with st.expander("Debug Information"):
        st.write(f"Processing {len(cleaned_row_ids)} row IDs")
        st.write(f"DataFrame has {len(df)} rows and {len(df.columns)} columns")
        st.write(f"Image-related columns present: {'matched_s3_key' in df.columns}, {'HAS_IMAGE' in df.columns}")

        # Count rows with s3 keys
        if 'matched_s3_key' in df.columns:
            s3_key_count = df['matched_s3_key'].notna().sum()
            st.write(f"Number of rows with S3 keys: {s3_key_count} ({s3_key_count / len(df) * 100:.1f}%)")

        # Show a few sample rows
        if not df.empty:
            sample_rows = []
            for row_id in cleaned_row_ids[:3]:
                if 'ROW_ID' in df.columns:
                    matching_rows = df[df['ROW_ID'] == row_id]
                    if not matching_rows.empty:
                        sample_row = matching_rows.iloc[0]
                        sample_data = {
                            'ROW_ID': row_id,
                            'has_photo_link': pd.notna(sample_row.get('PHOTO_LINK')),
                            'has_matched_s3_key': pd.notna(sample_row.get('matched_s3_key')),
                            'has_image_indicator': sample_row.get('HAS_IMAGE', False),
                            'year': sample_row.get('YEAR'),
                            'month': sample_row.get('MONTH_NUM'),
                            'template_code': sample_row.get('TEMPLATE_CODE'),
                            'account_code': sample_row.get('ACCOUNT_CODE')
                        }
                        sample_rows.append(sample_data)

            if sample_rows:
                st.write("Sample rows being processed:")
                st.table(pd.DataFrame(sample_rows))

    # Images by row
    images_by_row = {}

    # Process each row ID
    for row_id in cleaned_row_ids[:max_total]:  # Limit to max_total rows
        try:
            # Find the row in the DataFrame
            row = None
            if 'ROW_ID' in df.columns:
                matching_rows = df[df['ROW_ID'] == row_id]
                if not matching_rows.empty:
                    row = matching_rows.iloc[0]
            elif row_id in df.index:
                row = df.loc[row_id]

            if row is not None:
                # Find matching images using our simplified approach
                image_urls = find_matching_images(row)

                if image_urls:
                    images_by_row[row_id] = image_urls
            else:
                logger.warning(f"Row {row_id} not found in DataFrame")
        except Exception as e:
            logger.error(f"Error processing row {row_id}: {str(e)}")
            continue

    # Display images or fallback message
    if not images_by_row:
        st.info("No images found for the referenced examples. Will display sample images instead.")

        # Get a few sample images as fallback
        try:
            # Query for sample images directly
            from snowflake_connection import SnowflakeConnectionService

            query = """
            SELECT S3_KEY
            FROM INFORMATION_MART.ISA_S3_IMAGES
            WHERE S3_KEY IS NOT NULL
            LIMIT 6
            """

            sample_images_df = SnowflakeConnectionService.fetch_query_result(query)

            if not sample_images_df.empty:
                sample_urls = []
                bucket_name = f'vapafpdf-{os.getenv("FDF_ENV", "dev")}-ds-storage'

                for _, img_row in sample_images_df.iterrows():
                    image_url = get_signed_url(img_row['S3_KEY'], bucket_name)
                    if image_url:
                        sample_urls.append(image_url)

                if sample_urls:
                    # Display sample images
                    st.subheader("📷 Sample Images (Not Related to Report)")
                    cols = st.columns(3)
                    for i, url in enumerate(sample_urls):
                        if i < len(cols):
                            with cols[i]:
                                try:
                                    st.image(url, caption=f"Sample Image {i + 1}", use_column_width=True)
                                except Exception as e:
                                    st.error(f"Failed to load sample image: {e}")
                                    st.markdown(f"[View Image]({url})")
        except Exception as e:
            st.error(f"Failed to retrieve sample images: {e}")
    else:
        # Display found images
        st.subheader("📷 Images from Referenced Examples")

        # Create columns for grid display
        cols_per_row = 3
        rows_needed = (len(images_by_row) + cols_per_row - 1) // cols_per_row

        # Display images by row ID
        row_idx = 0
        for i in range(rows_needed):
            cols = st.columns(cols_per_row)

            for j in range(cols_per_row):
                if row_idx >= len(images_by_row):
                    break

                row_id = list(images_by_row.keys())[row_idx]
                image_urls = images_by_row[row_id]

                with cols[j]:
                    try:
                        st.image(
                            image_urls[0],  # Just show the first image
                            caption=f"Row ID: {row_id}",
                            use_column_width=True
                        )
                    except Exception as e:
                        st.error(f"Failed to load image for row {row_id}: {e}")
                        st.markdown(f"[View Image URL]({image_urls[0]})")

                row_idx += 1

    def display_report_with_images(df: pd.DataFrame, report: str, examples_ids: List[Union[int, str]]):
        """
        Display the generated report with associated images side by side.

        Args:
            df: DataFrame containing the data used for report generation
            report: The text content of the generated report
            examples_ids: List of row IDs referenced in the report
        """
        logger.info("Displaying report with images")

        # If no explicit example IDs provided, try to extract them from the report
        if not examples_ids:
            examples_ids = extract_row_ids_from_text(report)
            logger.info(f"Extracted {len(examples_ids)} row IDs from report text")

        # Create a two-column layout
        col_report, col_photos = st.columns([3, 2])

        with col_report:
            st.title("📊 Key Insights Report")

            with st.container():
                st.markdown("### Executive Summary")
                html_content = """
                    <div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px;'>
                    {content}
                    </div>
                    """.format(
                    content=report.replace("\n", "<br>")
                )
                st.markdown(html_content, unsafe_allow_html=True)

        with col_photos:
            st.title("🖼️ Examples & Photos")

            # Display images using our standalone function
            display_images_for_report(df, examples_ids, max_total=6)