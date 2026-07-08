# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# Author     ：Bo Wang
# File       : blobstore.py
# Time       ：2022/10/26 11:41
"""
# -*- encoding: utf-8 -*-
import glob
import logging
import threading

import boto3
import os
from botocore import UNSIGNED
from botocore.config import Config
import boto3
from urllib.parse import urlsplit
from typing import Tuple
import cv2
import os

s3_client_config = Config(s3={"addressing_style": "auto"}, signature_version=UNSIGNED)  # 固定配置，不可更改

s3_endpoint_url = "http://bs3-hb1.internal"  # 线上环境


# endpoint_url = "http://bs3-hb1.staging.kuaishou.com"         # Staging 环境


class BlobStoreClient(object):
    def __init__(self, bucket_name):
        self.s3_bucket = bucket_name
        self.s3_client = boto3.client("s3", endpoint_url=s3_endpoint_url, config=s3_client_config)

    # see: https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html#S3.Client.put_object
    def upload_binary_to_s3(self, file_path, key):
        try:
            binfile = open(file_path, 'rb')
            with open(file_path, 'rb') as binfile:
                self.s3_client.put_object(Bucket=self.s3_bucket, Body=binfile, Key=key)
                print('upload_succ:', key)
        except Exception as e:
            logging.error(e)

    # see: https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html#S3.Client.get_object
    def download_binary_from_s3(self, key, dest_key, bucket=""):
        try:
            real_bucket = bucket if len(bucket) > 0 else self.s3_bucket
            response = self.s3_client.get_object(Bucket=real_bucket, Key=key)
            # print(response)
            if response['ResponseMetadata']['HTTPStatusCode'] != 200:
                print(response)
                return False
            binfile = open(dest_key, 'wb')
            binfile.write(response["Body"].read())
            binfile.close()
            print('download and write succ:', key, dest_key)
        except Exception as e:
            logging.error(e)
            return False
        return True

    def batch_download_binary_from_s3(self, item_map):
        for k, v in item_map.items():
            self.download_binary_from_s3(k, v[0], v[1])

    def concurrency_download(self, item_map, batch_size=1):
        count = 0
        batch_list = []
        thread_list = []
        for k, v in item_map.items():
            if count % batch_size == 0:
                batch_list.append({})
            batch_list[-1].update({k: v})
            count += 1

        for batch in batch_list:
            thread_list.append(threading.Thread(target=self.batch_download_binary_from_s3, args=(batch, )))

        for t in thread_list:
            t.start()

        for t in thread_list:
            t.join()

    def object_exists(self, key):
        try:
            # response = self.s3_client.object_exists(Bucket=self.s3_bucket, Key=key)
            response = self.s3_client.head_object(Bucket=self.s3_bucket, Key=key)
            print(response)
            return True
        except Exception as e:
            logging.error(e)
            return False

    def get_object_bytes(self, key):
        try:
            real_bucket = self.s3_bucket
            response = self.s3_client.get_object(Bucket=real_bucket, Key=key)
            if response['ResponseMetadata']['HTTPStatusCode'] != 200:
                print(response)
                return None
            return response["Body"].read()
        except Exception as e:
            logging.error(e)
            return None




# see: https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html#S3.Client.get_object
# def get_meta_demo(self):
#    try:
#        response = client.get_object(Bucket=bucket, Key="test-python")
#        print(response["ContentType"])
#    except Exception as e:
#        logging.error(e)


def test_client():
    key = 'fullsds_45009408746.mp4'
    dest_key = 'test_wide_fullsds_45009408746.mp4'
    client = BlobStoreClient('ad-ad-material-union')
    if client.download_binary_from_s3(key, dest_key):
        here_direction = os.path.dirname(__file__)
        file_path = os.path.join(here_direction, dest_key)
        client.upload_binary_to_s3(file_path, dest_key)
        response = client.s3_client.get_object(Bucket=client.s3_bucket, Key=dest_key)
        if os.path.exists(file_path):
            os.remove(file_path)
        print(response)


def upload(filenames, output):
    client = BlobStoreClient('ad-nieuwland-material')
    for filename in filenames:
        key = os.path.splitext(os.path.basename(filename))[0]
        print(key)
        client.upload_binary_to_s3(filename, key)
        response = client.s3_client.get_object(Bucket=client.s3_bucket, Key=key)
        print(response)


def resource_decode(source, split_flag="_"):
    items = source.split(split_flag)
    db, tabel, key = items[0], items[1], split_flag.join(items[2:])
    return db, tabel, key

def download(resource_id, output):
    db, tabel, key = resource_decode(resource_id)
    client = BlobStoreClient(f'{db}-{tabel}')
    client.download_binary_from_s3(key, output)


if __name__ == '__main__':
    # test_client()
    for i in range(1,11):
        upload(glob.glob(f"screenshot_dataset_part{i}_kousiqi.tar.gz"), "")
    #upload(glob.glob("3202_replica_4o.lzh.png"), "")
    #download("ad-nieuwland_material_3202_replica_4o.lzh", "3202_replica_4o_new.png")
