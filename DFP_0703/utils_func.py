import re
import pandas as pd
from datetime import datetime


def clean_text(text):
    # Remove all non-printable characters except standard ASCII and common currency symbols
    text = re.sub(r"[^\x20-\x7E€£¥₹$]", " ", text)

    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)

    # Strip leading and trailing whitespace
    return text.strip()


def filter_date_range(df, date_column, from_date=None, to_date=None):
    """
    Filters and sorts a dataframe by a specified date column,
    returning rows within the specified date range.

    Args:
    - df (pd.DataFrame): The input dataframe.
    - date_column (str): The name of the column containing dates.
    - from_date (datetime, optional): The start date for filtering. Defaults to None.
    - to_date (datetime, optional): The end date for filtering. Defaults to None.

    Returns:
    - pd.DataFrame: A filtered and sorted dataframe.
    """
    # Convert the date column to datetime format

    # Assuming `from_date` or `to_date` is of type datetime.date
    from_date = datetime.combine(from_date, datetime.min.time())  # Convert to datetime
    to_date = datetime.combine(
        to_date, datetime.min.time()
    )  # Convert to datetime (if applicable)

    df[date_column] = pd.to_datetime(df[date_column])

    # Order by the date column
    df = df.sort_values(by=date_column)

    # Apply filtering based on the provided dates
    if from_date is not None:
        df = df[df[date_column] >= from_date]
    if to_date is not None:
        df = df[df[date_column] <= to_date]

    return df

