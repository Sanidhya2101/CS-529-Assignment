from sklearn.decomposition import PCA,TruncatedSVD
import pandas as pd

def pca_reduction(X_train,X_test,y_train):

    pca = PCA(n_components=200)
    
    train_pca = pd.DataFrame(pca.fit_transform(X_train,y_train),columns=pca.get_feature_names_out())
    test_pca = pd.DataFrame(pca.transform(X_test),columns=pca.get_feature_names_out())

    return (train_pca,test_pca)

def lsi_reduction(X_train,X_test,y_train):

    svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)

    train_lsi = pd.DataFrame(svd.fit_transform(X_train,y_train),columns=svd.get_feature_names_out())
    test_lsi = pd.DataFrame(svd.transform(X_test),columns=svd.get_feature_names_out())

    return (train_lsi,test_lsi)
