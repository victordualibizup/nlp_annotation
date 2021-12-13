import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from nlp_annotation.config.core import config

pipeline_list = [
    ("count_vectorizer", CountVectorizer(
        tokenizer=word_tokenize,
        token_pattern=config.model_config.token_pattern,
        ngram_range=(
            config.model_config.ngram_range_min,
            config.model_config.ngram_range_max
        )
    )
     ),

    ("logistic_regressor",
     LogisticRegression(solver=config.model_config.logistic_regression_solver)
     )
]

model_pipeline = Pipeline(pipeline_list)
