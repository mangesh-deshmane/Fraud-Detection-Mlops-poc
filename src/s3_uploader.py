import boto3
import os
import argparse
import logging
import sys
from botocore.exceptions import NoCredentialsError, ClientError

def setup_logging():
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        filename='logs/s3_upload.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def upload_to_s3(local_folder, bucket_name, s3_folder):
    s3_client = boto3.client('s3')
    
    try:
        logging.info(f"Starting upload from {local_folder} to s3://{bucket_name}/{s3_folder}")
        
        if not os.path.exists(local_folder):
             logging.error(f"Local folder not found: {local_folder}")
             sys.exit(1)

        for root, dirs, files in os.walk(local_folder):
            for file in files:
                local_path = os.path.join(root, file)
                
                # Calculate relative path to maintain folder structure in S3
                relative_path = os.path.relpath(local_path, local_folder)
                s3_path = os.path.join(s3_folder, relative_path)
                
                logging.info(f"Uploading {local_path} to {s3_path}")
                s3_client.upload_file(local_path, bucket_name, s3_path)
        
        logging.info("Upload completed successfully.")

    except NoCredentialsError:
        logging.error("Credentials not available. Please configure AWS credentials.")
        sys.exit(1)
    except ClientError as e:
        logging.error(f"S3 Client Error: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Upload failed: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="S3 Uploader Script")
    parser.add_argument('--bucket', type=str, required=True, help='S3 Bucket Name')
    parser.add_argument('--local_folder', type=str, default='artifacts', help='Local folder to upload')
    parser.add_argument('--s3_folder', type=str, default='fraud-detection/artifacts', help='Destination folder in S3')
    args = parser.parse_args()

    setup_logging()
    upload_to_s3(args.local_folder, args.bucket, args.s3_folder)

if __name__ == "__main__":
    main()
