import requests
import json
import os
import numpy as np
#import tensorflow as tf
import pandas as pd
import h5py
import sys
import datetime
import argparse
#import logging

os.environ["HDF5_USE_FILE_LOCKING"]='FALSE'

import boto3
import s3fs

s3 = boto3.resource('s3')

def read_data(filename, rank=0, size=1, end=None, dtype= np.float32, MEAN= 33.44, SCALE = 47.54):
    xkeys= ['IN']
    y_keys= ['OUT']
    s= np.s_[rank:end:size]

    #s3= s3fs.S3FileSystem()
    #with s3.open(filename, 'rb') as s3File:
    with h5py.File(s3File, 'r') as hf:
        IN= hf ['IN'][s]
        OUT= hf ['OUT'][s]
    IN = (IN.astype(dtype)-MEAN )/SCALE
    OUT= (OUT.astype(dtype)-MEAN )/SCALE
    return IN, OUT

def handler(event, context):
    """
    Runs data processing scripts to extract training set from SEVIR
    """
    #logger = logging.getLogger(__name__)
    #logger.info('making final data set from raw data')
    #tst_generator = get_nowcast_test_generator(sevir_catalog=DATA_CATALOG,
                                              # sevir_location=DATA_sevir)

    #logger.info('Reading/writing testing data to %s' % ('%s/nowcast_testing.h5' % DATA_interim))
    #read_write_chunks('%s/nowcast_testing.h5' % DATA_interim,tst_generator,20)
    x_test,y_test= read_data('s3://bucket-satellite/data/nowcast_testing000.h5', end=10)
    print(x_test.shape)
    print(y_test.shape)
    return {
        'headers': {'Content-Type': 'sevir Ingestion Pipeline'},
        'statusCode': 200,
        'body': json.dumps({"message": "x_test, y_test generated",
                           "event": event})
    }