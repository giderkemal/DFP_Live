import datetime
import pandas as pd
import sys
import os

from model import Classification

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from snowflake_connection import SnowflakeConnectionService

env = os.getenv('FDF_ENV', 'dev')


def main():
    classif = Classification()

    if env == 'dev':
        query = '''
        select top 1000
        df.task_code,
        df.form_type,
        df.field_intelligence,
        df.field_intelligence_translated,
        df.class
        
        from information_mart.df_passport_final df
        left join information_mart.df_passport_classification cl on df.task_code = cl.task_code
        where cl.class is null
        '''
    else:
        query = '''
        select top 1000
        df.task_code,
        df.form_type,
        df.field_intelligence,
        df.field_intelligence_translated,
        df.class

        from integration.v_int_fact_df_passport df
        where df.class is null
        '''

    df = SnowflakeConnectionService.fetch_query_result(query)

    classified_df = classif.process_dataframe(df, num_workers=10)

    classified_df['LAST_MODIFIED_DATETIME'] = pd.to_datetime(datetime.datetime.now())
    classified_df['LAST_MODIFIED_DATETIME'] = classified_df['LAST_MODIFIED_DATETIME'].dt.strftime('%Y-%m-%d %H:%M:%S')

    SnowflakeConnectionService.save_to_snowflake(
        df=classified_df,
        table_name='DF_PASSPORT_CLASSIFICATION',
        schema='INFORMATION_MART',
        overwrite=False,
    )


if __name__ == "__main__":
    main()
