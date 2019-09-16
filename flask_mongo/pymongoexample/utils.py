from pyspark.sql.functions import regexp_replace, lower, monotonically_increasing_id, lit
from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import NGram
from .extensions import mongo
from flask import abort

#Function that create a Spark dataframe and remove special chars
def build_dataframe(spark):
    files_collection = mongo.db.files
    files = list(files_collection.find())
    if(len(files) == 0):
        abort(404, description="Favor inserir pelo menos um documento")
    dataframe = [tuple([files[i]['name'], files[i]['text']]) for i in range(0,len(files))]
    df = spark.createDataFrame(dataframe, ['name', 'text'])
    return df

#Function that returns the vocabulary of the documents
def vocab_return(df):
    pre_data = pre_process_data(df)
    count_model = count_vectorizer(pre_data)
    dict_vocab = {
        "vocabulary": count_model.vocabulary
    }
    return dict_vocab

#Function that returns the ngram vocabulary of the documents
def ngram_vocab_return(df):
    pre_data = pre_process_data(df)
    ngram_data = ngram(pre_data)
    count_model = count_vectorizer(ngram_data,"ngrams")
    dict_ngram = {
        "ngram_vocabulary": count_model.vocabulary
    }
    return dict_ngram

#Function that returns the vocabulary vector of each document
def vocab_vector_return(df):
    pre_data = pre_process_data(df)
    count_model = count_vectorizer(pre_data)
    return build_vector(count_model, pre_data)

#Function that returns the ngram vocabulary vector of each document
def ngram_vector_return(df):
    pre_data = pre_process_data(df)
    ngram_data = ngram(pre_data)
    count_model = count_vectorizer(ngram_data,"ngrams")
    return build_vector(count_model, ngram_data)

#Function to pre process data, tokenize text and remove stop words.
def pre_process_data(df):
    df_collumn = df.withColumn("text", regexp_replace(lower(df["text"]), "[$&+,:;=?@#|'<>.-^*()%!]", ""))
    df_without = df_collumn.withColumn("text", regexp_replace(lower(df_collumn["text"]), "-", " "))
    df_read = df_without.select('*').withColumn("id", monotonically_increasing_id())
    # Tokenize data
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    df_tokenized = tokenizer.transform(df_read)
    #Remove Stop Words
    language = "portuguese"
    remover = StopWordsRemover(inputCol="words", outputCol="filtered",
                               stopWords=StopWordsRemover.loadDefaultStopWords(language))
    df_clean = remover.transform(df_tokenized)
    #Return dataframe
    return df_clean

#Function that makes a count_vectorizer model that allows to get the vocabulary
def count_vectorizer(df_clean, col_tobe_vec="filtered"):
    cv = CountVectorizer(inputCol=col_tobe_vec, outputCol="features")
    model = cv.fit(df_clean)
    return model

#Function that create the ngram collumn
def ngram(df_clean):
    ngram = NGram(n=2, inputCol="filtered", outputCol="ngrams")
    ngramDataFrame = ngram.transform(df_clean)
    return ngramDataFrame

#Function that build a vector with the word frequency for each document.
def build_vector(model, pre_data):
    result = model.transform(pre_data)
    name_list = result.select('name').collect()
    names = [row.name for row in name_list]
    selected = result.select('features')
    dict_vector = {}
    #for each file create the vector of frequency
    for ind, x in enumerate(names):
        dict_vector[x] = list(map(int, selected.collect()[ind][0].toArray()))
    return dict_vector