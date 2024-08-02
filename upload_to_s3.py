import boto3
import os

aws_access_key_id = 'AKIAYAV34GOXJ2BZLK66'
aws_secret_access_key = '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'
region_name = 'ap-south-1'
bucket_name = 'pose-estimation-internship'

# Establishing a connection to AWS S3
s3 = boto3.resource(
    's3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=region_name
)

def upload_directory(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            local_file_path = os.path.join(root, file)
            s3_key = os.path.relpath(local_file_path, path)
            s3_key = s3_key.replace('\\', '/') 
            try:
                s3.meta.client.upload_file(local_file_path, bucket_name, s3_key)
                print(f'{file} uploaded successfully to {bucket_name} as {s3_key}.')
            except Exception as e:
                print(f'Error uploading {file} to {bucket_name}: {e}')

# Example usage: Upload all files in a local directory recursively
local_directory = r'D:\Infosys_AI_Model\Downloaded Videos'  # Replace with the path to your local directory
upload_directory(local_directory)
