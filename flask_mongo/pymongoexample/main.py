from flask import Blueprint, request
import requests
import sys
import json
from .extensions import mongo 
from .utils import vocab_return, ngram_vocab_return, vocab_vector_return, ngram_vector_return, build_dataframe

main = Blueprint('main', __name__)
import findspark
findspark.init()
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('cin').getOrCreate()
# from requests.exceptions import HTTPError
# from pyspark.sql.types import StructType, StructField, StringType

@main.route('/postFiles', methods=['POST'])
def index():
    files_collection = mongo.db.files
    data = request.json
    response = requests.get(data['link']).text
    filename = data['link'].rsplit("/", 1)
    files_collection.insert({'name': filename[len(filename)-1], 'text': response})
    return 'Added File'

@main.route('/postText', methods=['POST'])
def indexText():
    files_collection = mongo.db.files
    data = request.json
    files_collection.insert({'name': data['name'], 'text': data['text']})
    return 'Added Text'


@main.route('/getVocabulary')
def vocab():
    df = build_dataframe(spark)
    vocab_data = vocab_return(df)
    return vocab_data

@main.route('/getNgramVocabulary')
def ngram_vocab():
    df = build_dataframe(spark)
    ngram_data = ngram_vocab_return(df)
    print(ngram_data, file=sys.stderr)
    return ngram_data

@main.route('/getVector')
def vector():
    df = build_dataframe(spark)
    vector_data = vocab_vector_return(df)
    return vector_data

@main.route('/getNgramVector')
def ngram_vec():
    df = build_dataframe(spark)
    ngram_vector_data = ngram_vector_return(df)
    return ngram_vector_data