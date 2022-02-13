from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from scipy.sparse import hstack

def create_dataframe(filename):
    df = pd.read_csv(filename)
    df.columns = [c.replace(' ', '_') for c in df.columns]
    return df



def gettfidf(train_df,test_df):

    body_text_vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, stop_words='english')
    headline_vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, stop_words='english')

    train_body_tfidf = body_text_vectorizer.fit_transform(train_df['articleBody'])
    train_headline_tfidf = headline_vectorizer.fit_transform(train_df['Headline'])

    test_body_tfidf = body_text_vectorizer.transform(test_df['articleBody'])
    test_headline_tfidf = headline_vectorizer.transform(test_df['Headline'])

    train_features = hstack([train_body_tfidf, train_headline_tfidf])
    test_features = hstack([test_body_tfidf, test_headline_tfidf])


    return (train_features,test_features)
