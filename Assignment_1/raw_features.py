from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


def create_dataframe(filename):
    df = pd.read_csv(filename)
    df.columns = [c.replace(' ', '_') for c in df.columns]
    return df

def gettfidf(train_df,test_df):

    vectorizer = TfidfVectorizer(ngram_range=(1,2),lowercase=True,stop_words='english',max_features=2500)
        

    train_features = pd.DataFrame(vectorizer.fit_transform(train_df['Headline+Body']).todense(),columns=vectorizer.get_feature_names_out())
    test_features = pd.DataFrame(vectorizer.transform(test_df['Headline+Body']).todense(),columns=vectorizer.get_feature_names_out())

    return (train_features,test_features)
