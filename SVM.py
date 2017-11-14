from sklearn import svm

def my_svm(X_train, Y_train):
    model = svm.SVC(kernel='linear').fit(X_train, Y_train)
    return model

def decisionfunction(model):
    dec = model.decision_function()

def predict(model, X_test):
    pred = model.predict(X_test)
    return pred
