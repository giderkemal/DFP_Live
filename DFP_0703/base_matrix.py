import pandas as pd
import streamlit as st

from snowflake_connection import SnowflakeConnectionService


class BaseMatrixService:
    @staticmethod
    def reorder_columns(df):
        """Reorders the columns of a DataFrame based on a predefined order."""
        custom_column_order = [
            "SUBMISSION_DATETIME",
            "FORM_TYPE",
            "VP_REGION_NAME",
            "DF_MARKET_NAME",
            "LOCATION_NAME",
            "PMI_PRODUCT_NAME",
            "PRODUCT_CATEGORY_NAME",
            "BRAND_NAME",
            "BRAND_NAME_FROM",
            "BRAND_NAME_TO",
            "TMO_NAME",
            "FIELD_INTELLIGENCE",
            "FIELD_INTELLIGENCE_TRANSLATED",
            "POV_ID",
            "CLASS",
        ]

        # Keep only existing columns in the specified order
        reordered_df = df[[col for col in custom_column_order if col in df.columns]]

        return reordered_df

    @staticmethod
    def reorder_rows(df):
        """Reorders the rows of the Base Matrix DataFrame based on Date."""

        # Sort by Date
        reordered_df = df.sort_values(by="SUBMISSION_DATETIME", ascending=False)
        # Reset index
        reordered_df.reset_index(drop=True, inplace=True)

        return reordered_df

    @staticmethod
    @st.cache_data(ttl=86400)
    def create_base_matrix(scope):
        base_matrix = SnowflakeConnectionService.get_raw_base_matrix()

        renamed_scope = BaseMatrixService.rename_scope(scope)

        transformed_scope = BaseMatrixService.adjust_base_matrix_scope(renamed_scope)

        scope_list = transformed_scope.where(pd.notna(transformed_scope), None).to_dict(
            orient="records"
        )

        filtered_df = BaseMatrixService.filter_base_matrix(base_matrix, scope_list)

        # Reorder columns
        ordered_df = BaseMatrixService.reorder_columns(filtered_df)

        return ordered_df, ordered_df.shape[0]

    @staticmethod
    def rename_scope(scope: pd.DataFrame):
        rename_dict = {
            "Date range": "DATE_RANGE",
            "Feedback class": "CLASS",
            "Form type": "FORM_TYPE",
            "Region": "VP_REGION_NAME",
            "Market": "DF_MARKET_NAME",
            "Location": "LOCATION_NAME",
            "TMO": "TMO_NAME",
            "Product Category": "PRODUCT_CATEGORY_NAME",
            "PMI Product": "PMI_PRODUCT_NAME",
            "Brand": "BRAND_NAME",
            "Switch From": "BRAND_NAME_FROM",
            "Switch To": "BRAND_NAME_TO",
            "Answer Category": "ANSWER_CATEGORY",
        }

        scope_renamed = scope.rename(columns=rename_dict)

        return scope_renamed

    @staticmethod
    def adjust_base_matrix_scope(base_matrix_scope):
        transformed_df = base_matrix_scope.copy()
        for col in transformed_df.columns:
            if col == "DATE_RANGE":
                transformed_df[col] = transformed_df[col].apply(
                    lambda x: x.split(" — ") if isinstance(x, str) else x
                )
            else:
                transformed_df[col] = transformed_df[col].apply(
                    lambda x: x.split(", ") if isinstance(x, str) else x
                )

        return transformed_df

    @staticmethod
    def form_where_condition(scope):
        """
        Used to double_check filtering values against Snowflake query
        """

        or_clauses = []

        for condition in scope:
            and_clauses = []
            for key, value in condition.items():
                if value is None or value == ["All"]:
                    continue
                elif key == "DATE_RANGE":
                    and_clauses.append(
                        f"(submission_datetime between '{value[0]}' and '{value[1]}')"
                    )
                elif key == "ANSWER_CATEGORY":
                    values = "({})".format(', '.join(f"'{v}'" for v in value))
                    and_clauses.append(f"product_category_name in {values}")
                elif isinstance(value, list):
                    values = "({})".format(', '.join(f"'{v}'" for v in value))
                    and_clauses.append(f"{key} in {values}")

            or_clauses.append(f"({' and '.join(and_clauses)})")

        return "where" + " or ".join(or_clauses)

    @staticmethod
    def filter_base_matrix(df, scope):
        df["SUBMISSION_DATETIME"] = pd.to_datetime(df["SUBMISSION_DATETIME"])
        final_mask = pd.Series(False, index=df.index)

        for condition in scope:
            row_mask = pd.Series(True, index=df.index)

            for key, value in condition.items():
                if value is None or value == ["All"]:
                    continue

                elif key == "DATE_RANGE":
                    row_mask &= (
                        df["SUBMISSION_DATETIME"] >= pd.to_datetime(value[0])
                    ) & (df["SUBMISSION_DATETIME"] <= pd.to_datetime(value[1]))

                elif key == "ANSWER_CATEGORY":
                    row_mask &= df["PRODUCT_CATEGORY_NAME"].isin(value)

                elif isinstance(value, list):
                    row_mask &= df[key].isin(value)

            final_mask |= row_mask

        return df[final_mask]
