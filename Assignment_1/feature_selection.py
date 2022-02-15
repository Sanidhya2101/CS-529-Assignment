from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2
import pandas as pd


def mutual_info(X_train,y_train,X_test):
    
    mi = SelectKBest(mutual_info_classif,k=500)

    
    train_mi = pd.DataFrame(mi.fit_transform(X_train,y_train),columns=mi.get_feature_names_out())
    test_mi = pd.DataFrame(mi.transform(X_test),columns=mi.get_feature_names_out())


    return (train_mi,test_mi)


def chi2_features(X_train,y_train,X_test):
    
    chi = SelectKBest(chi2,k=500)

    train_chi = pd.DataFrame(chi.fit_transform(X_train,y_train),columns=chi.get_feature_names_out())
    test_chi = pd.DataFrame(chi.transform(X_test),columns=chi.get_feature_names_out())


    return (train_chi,test_chi)
