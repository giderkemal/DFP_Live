import os
import json
import boto3
import pandas as pd


class S3ClientManager:
    env = os.getenv('FDF_ENV', 'dev')
    _client = None

    @classmethod
    def get_s3_credentials(cls):
        secrets_client = boto3.client(
            'secretsmanager',
            # aws_access_key_id='',
            # aws_secret_access_key='',
            region_name='eu-west-1',
        )

        response = secrets_client.get_secret_value(SecretId=f'vapafpdf-{cls.env}-image_search_user_keys')

        secret = response['SecretString']
        secret_dict = json.loads(secret)

        aws_access_key_id = secret_dict['aws_access_key_id']
        aws_secret_access_key = secret_dict['aws_secret_access_key']

        credentials = {
            'aws_access_key_id': aws_access_key_id,
            'aws_secret_access_key': aws_secret_access_key,
        }

        return credentials

    @classmethod
    def get_s3_client(cls):
        if cls._client is None:
            credentials = S3ClientManager.get_s3_credentials()

            access_key = credentials['aws_access_key_id']
            secret_access_key = credentials['aws_secret_access_key']

            cls._client = boto3.client(
                's3',
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_access_key,
            )

        return cls._client


def get_signed_url(
    s3_key: str,
    bucket_name: str,
    expiry: int = 360
) -> str:
    if pd.isna(s3_key) or not s3_key:
        return None

    s3_client = S3ClientManager.get_s3_client()

    return s3_client.generate_presigned_url(
        'get_object',
        Params={'Bucket': bucket_name, 'Key': s3_key},
        ExpiresIn=expiry,
    )
