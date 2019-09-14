from flask import Blueprint
import sys
from .extensions import mongo 

main = Blueprint('main', __name__)

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
    return 'vocab'

@main.route('/getNgramVocabulary')
def ngram_vocab():
    files_collection = mongo.db.files
    files = files_collection.find({})
    print(files[0], file=sys.stderr)
    return 'Ngram_vocab'

@main.route('/getVector')
def vector():
    files_collection = mongo.db.files
    files = files_collection.find({})
    print(files[0], file=sys.stderr)
    return 'vector'

@main.route('/getNgramVector')
def ngram_vec():
    files_collection = mongo.db.files
    files = files_collection.find({})
    print(files[0], file=sys.stderr)
    return 'Ngram vector'