from flask import Blueprint, request, abort
import requests
import sys
from .extensions import mongo
from .utils import vocab_return, ngram_vocab_return, vocab_vector_return, ngram_vector_return, build_dataframe
main = Blueprint('main', __name__)
import findspark
findspark.init()
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('cin').getOrCreate()

#Route to post files, here you should make a request with a link
@main.route('/postFiles', methods=['POST'])
def index():
    files_collection = mongo.db.files
    data = request.json
    if(data['link']):
        response = requests.get(data['link'])
        response.encoding = 'utf-8'
        if(response.status_code == 200):
            textfile = response.text
            print(response.encoding, file=sys.stderr)
            filename = data['link'].rsplit("/", 1)
            files_collection.insert({'name': filename[len(filename)-1], 'text': textfile})
        else:
            abort(404, description= "O link não é valido")
    return 'Arquivo Adicionado'

#Route to post text, here you should post a request with name and text attributes
@main.route('/postText', methods=['POST'])
def indexText():
    files_collection = mongo.db.files
    data = request.json
    if(data['name'] and data['text']):
        files_collection.insert({'name': data['name'], 'text': data['text']})
    return 'Texto adicionado'

#Route to get the vocabulary from all documents
@main.route('/getVocabulary', methods=['GET'])
def vocab():
    df = build_dataframe(spark)
    vocab_data = vocab_return(df)
    return vocab_data

#Route to get the Ngram Vocabulary from all documents
@main.route('/getNgramVocabulary', methods=['GET'])
def ngram_vocab():
    df = build_dataframe(spark)
    ngram_data = ngram_vocab_return(df)
    return ngram_data

#Route to get the vector of the vocabulary with the word frequency per file
@main.route('/getVector', methods=['GET'])
def vector():
    df = build_dataframe(spark)
    vector_data = vocab_vector_return(df)
    return vector_data

#Route to get Ngram Vector with ngram frequency per file
@main.route('/getNgramVector', methods=['GET'])
def ngram_vec():
    df = build_dataframe(spark)
    ngram_vector_data = ngram_vector_return(df)
    return ngram_vector_data

#Route to clear all files
@main.route('/clearFiles', methods=['DELETE'])
def clear():
    files_collection = mongo.db.files
    files_collection.delete_many({})
    return 'files cleaned'