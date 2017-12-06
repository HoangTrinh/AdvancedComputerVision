import tensorflow as tf
import numpy as np
import Knn
import matplotlib.pyplot as plt
import SVM
import os
import support_function
import Bayes

Root = 'db/'
for data_direct in os.listdir(Root):
    folder_direct = Root + data_direct + '/'
    X_train, Y_train = support_function.get_features_labels_from_folder(folder_direct,'train', 'vgg19')
    X_test, Y_test = support_function.get_features_labels_from_folder(folder_direct,'dev', 'vgg19')

    # #SVM run + score ( 0.9 )
    # model = SVM.my_svm(X_train, Y_train)
    # score = model.score(X_test, Y_test)

    # # Knn run + score (0.64 )
    # model  = Knn.sci_knn(X_train,Y_train, 5)
    # score = model.score(X_test, Y_test)
    # print(score)

    # #native Bayes + score (0.436)
    # model = Bayes.bayes_with_nativeG(X_train, Y_train)
    # score = model.score(X_test, Y_test)
    # print(score)