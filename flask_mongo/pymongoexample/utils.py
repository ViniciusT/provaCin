from pyspark.sql.functions import regexp_replace, lower, monotonically_increasing_id, lit
from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import NGram
import sys
from .extensions import mongo


def build_dataframe(spark):
    files_collection = mongo.db.files
    files = list(files_collection.find())
    dataframe = [tuple([files[i]['name'], files[i]['text']]) for i in range(0,len(files))]
    df = spark.createDataFrame(dataframe, ['name', 'text'])
    return df

def vocab_return(df):
    pre_data = pre_process_data(df)
    count_model = count_vectorizer(pre_data)
    dict_vocab = {
        "vocabulary": count_model.vocabulary
    }
    return dict_vocab

def ngram_vocab_return(df):
    pre_data = pre_process_data(df)
    ngram_data = ngram(pre_data)
    count_model = count_vectorizer(ngram_data,"ngrams")
    dict_ngram = {
        "ngram_vocabulary": count_model.vocabulary
    }
    return dict_ngram

def vocab_vector_return(df):
    pre_data = pre_process_data(df)
    count_model = count_vectorizer(pre_data)
    return build_vector(count_model, pre_data)

def ngram_vector_return(df):
    pre_data = pre_process_data(df)
    ngram_data = ngram(pre_data)
    count_model = count_vectorizer(ngram_data,"ngrams")
    return build_vector(count_model, ngram_data)

def pre_process_data(df):
    df_collumn = df.withColumn("text", regexp_replace(lower(df["text"]), "[$&+,:;=?@#|'<>.-^*()%!]", ""))
    df_read = df_collumn.select('*').withColumn("id", monotonically_increasing_id())
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

def count_vectorizer(df_clean, col_tobe_vec="filtered"):
    cv = CountVectorizer(inputCol=col_tobe_vec, outputCol="features")
    model = cv.fit(df_clean)
    return model

def ngram(df_clean):
    ngram = NGram(n=2, inputCol="filtered", outputCol="ngrams")
    ngramDataFrame = ngram.transform(df_clean)
    return ngramDataFrame

def build_vector(model, pre_data):
    print(pre_data.select("ngrams").show(truncate=False), file=sys.stderr)
    result = model.transform(pre_data)
    name_list = result.select('name').collect()
    names = [row.name for row in name_list]
    selected = result.select('features')
    dict_vector = {}
    for ind, x in enumerate(names):
        dict_vector[x] = list(map(int, selected.collect()[ind][0].toArray()))
    return dict_vector