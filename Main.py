import Knn
import SVM
import os
import support_function
import Bayes

Root = 'db/'

avg_svm_score = 0
avg_knn_score = 0
avg_bayes_score = 0

for data_direct in os.listdir(Root):
    print('dataset: ', data_direct)
    folder_direct = Root + data_direct + '/'
    X_train, Y_train = support_function.get_features_labels_from_folder(folder_direct,'train', 'vgg19')
    X_test, Y_test = support_function.get_features_labels_from_folder(folder_direct,'dev', 'vgg19')

    #SVM run + score ( 0.9 )
    model = SVM.my_svm(X_train, Y_train)
    score = model.score(X_test, Y_test)
    avg_svm_score += score
    print('svm', score)

    # Knn run + score (0.64 )
    model  = Knn.sci_knn(X_train,Y_train, 5)
    score = model.score(X_test, Y_test)
    avg_knn_score += score
    print('knn:',score)

    #native Bayes + score (0.436)
    model = Bayes.bayes_with_nativeG(X_train, Y_train)
    score = model.score(X_test, Y_test)
    avg_bayes_score += score
    print('bayes', score)

print('Mean svm score: %0.2f'%(avg_svm_score/len(os.listdir(Root))))
print('Mean knn score: %0.2f'%(avg_knn_score/len(os.listdir(Root))))
print('Mean bayes score: %0.2f'%(avg_bayes_score/len(os.listdir(Root))))