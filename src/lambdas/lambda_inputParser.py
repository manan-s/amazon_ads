import os
import sys
import json
import pickle
import uuid
import boto3
import logging
from urllib.parse import unquote_plus
from botocore.exceptions import ClientError, NoCredentialsError

s3_client = boto3.client('s3')

#Following function is just for testing locally hence can be ignored
def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload (A pickle file)
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name

    # Upload the file
    s3_client = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    except FileNotFoundError:
        print("The file was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False
    return True

def create_pickle(json_file_path):
    """Create a pickle file from the JSON data

    :param json_file_name: STRING, The name of the JSON file
    :param ASIN: STRING, The ASIN of the product, whose pricing data is acquired
    :return: STRING, The file name of pickle file created
    """

    with open(json_file_path, 'r') as json_file:
        json_data = json.load(json_file)

    data = {}
    data['timestamp'] = []
    data['value'] = []

    queryASIN = list(json_data.keys())[0]
    dataList = json_data[queryASIN]
    numPoints = len(dataList)

    for count in range(numPoints):
        tempDict = dataList[count]
        timestamp = tempDict["TimeStamp"]
        price = tempDict["Current_Price"]
        data['timestamp'].append(timestamp)
        data['value'].append(price)

    pickle_file_name = 'data.pkl'

    with open(pickle_file_name, 'wb') as f:
        pickle.dump(data, f)

    return pickle_file_name

#Uncomment following while working on local machine
pickle_file_name = create_pickle("data.json")

def lambda_handler(event, context):
    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        key = unquote_plus(record['s3']['object']['key'])
        tmpkey = key.replace('/', '')
        download_path = '/tmp/{}{}'.format(uuid.uuid4(), tmpkey)
        
        s3_client.download_file(bucket, key, download_path)
        pickle_file = create_pickle(download_path)
        upload_path = '/tmp/{}'.format(pickle_file)
        s3_client.upload_file(upload_path, 'ads-data-store')