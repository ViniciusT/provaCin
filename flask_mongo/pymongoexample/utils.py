from pyspark.sql.functions import regexp_replace, lower, monotonically_increasing_id, lit
from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import NGram

def vocab_return(spark, links):
    pre_data = pre_process_data(spark, links)
    count_model = count_vectorizer(pre_data)
    dict_vocab = {
        "vocabulary": count_model.vocabulary
    }
    return dict_vocab

def ngram_vocab_return(spark, links):
    pre_data = pre_process_data(spark, links)
    ngram_data = ngram(pre_data)
    count_model = count_vectorizer(ngram_data,"ngrams")
    dict_ngram = {
        "ngram_vocabulary": count_model.vocabulary
    }
    return dict_ngram

def vocab_vector_return(spark, links):
    pre_data = pre_process_data(spark, links)
    count_model = count_vectorizer(pre_data)
    return build_vector(count_model, pre_data, links)

def ngram_vector_return(spark, links):
    pre_data = pre_process_data(spark, links)
    ngram_data = ngram(pre_data)
    count_model = count_vectorizer(ngram_data,"ngrams")
    return build_vector(count_model, ngram_data, links)

def pre_process_data(spark, links):
    df = spark.read.option("encoding", "ISO-8859-1").text(['../data.txt', '../class.txt', '../newtest.txt'])
    df_collumn = df.withColumn("value", regexp_replace(lower(df["value"]), "[$&+,:;=?@#|'<>.-^*()%!]", ""))
    df_read = df_collumn.select('*').withColumn("id", monotonically_increasing_id())
    # Tokenize data
    tokenizer = Tokenizer(inputCol="value", outputCol="words")
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

def build_vector(model, pre_data, links):
    result = model.transform(pre_data)
    selected = result.select('features')
    dict_vector = {}
    for ind, x in enumerate(links):
        dict_vector[links[ind]] = list(map(int, selected.collect()[ind][0].toArray()))
    return dict_vector