from sklearn.naive_bayes import MultinomialNB

def bayes_with_nativeG(X_train, Y_train):
    model = MultinomialNB().fit(X_train,Y_train)
    return model