from flask import Blueprint
import sys
from .extensions import mongo 
from .utils import vocab_return, ngram_vocab_return, vocab_vector_return, ngram_vector_return

main = Blueprint('main', __name__)

import findspark
findspark.init()
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('cin').getOrCreate()
from pyspark.sql.functions import regexp_replace, lower, monotonically_increasing_id, lit
from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import CountVectorizer


@main.route('/postFiles')
def index():
    files_collection = mongo.db.files
    files_collection.insert({'link' : 'somelink2'})
    return '<h1>Added a User!</h1>'

@main.route('/getVocabulary')
def vocab():
    files_collection = mongo.db.files
    files = files_collection.find({})
    print(files[0]['link'], file=sys.stderr)
    vocab_data = vocab_return(spark, files[0]['link'])
    print(vocab_data, file=sys.stderr)
    return vocab_data

@main.route('/getNgramVocabulary')
def ngram_vocab():
    files_collection = mongo.db.files
    files = files_collection.find({})
    print(files[0], file=sys.stderr)
    ngram_data = ngram_vocab_return(spark, files[0]['link'])
    print(ngram_data, file=sys.stderr)
    return ngram_data

@main.route('/getVector')
def vector():
    files_collection = mongo.db.files
    files = files_collection.find({})
    print(files[0], file=sys.stderr)
    links = ['data.txt', 'class.txt', 'newtest.txt']
    vector_data = vocab_vector_return(spark, links)
    print(vector_data, file=sys.stderr)
    return vector_data

@main.route('/getNgramVector')
def ngram_vec():
    files_collection = mongo.db.files
    files = files_collection.find({})
    print(files[0], file=sys.stderr)
    links = ['data.txt', 'class.txt', 'newtest.txt']
    ngram_vector_data = ngram_vector_return(spark, links)
    print(ngram_vector_data, file=sys.stderr)
    return ngram_vector_data