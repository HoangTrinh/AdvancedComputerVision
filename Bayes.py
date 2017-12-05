from sklearn.naive_bayes import GaussianNB

def bayes_with_nativeG(X_train, Y_train):
    model = GaussianNB().fit(X_train,Y_train)
    return model