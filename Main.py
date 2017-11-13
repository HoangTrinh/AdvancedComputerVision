import tensorflow as tf
import numpy as np
import Knn

import matplotlib.pyplot as plt
plt.style.use('ggplot')
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (10.0, 5.0)


# random train data in 2D
num_points_each_cluster = 100
mu1 = [-0.4, 3]
covar1 = [[1.3,0],[0,1]]
mu2 = [0.5, 0.75]
covar2 = [[2.2,1.2],[1.8,2.1]]
X1 = np.random.multivariate_normal(mu1, covar1, num_points_each_cluster)
X2 = np.random.multivariate_normal(mu2, covar2, num_points_each_cluster)

y1 = np.ones(num_points_each_cluster)
y2 = np.zeros(num_points_each_cluster)

# Plot the train_data
# plt.plot( X1[:, 0], X1[:,1], 'ro', label='class 1')
# plt.plot(X2[:, 0], X2[:,1], 'bo', label='class 0')
# plt.legend(loc='best')
# plt.show()


X_train = np.vstack((X1, X2))
Y_train = np.hstack((y1, y2))

X_train_tf = tf.constant(X_train)
Y_train_tf = tf.constant(Y_train)

# Test generate

while(1):
    a = input('x_axis (-100 for existing): ')
    if a == '-100':
        break
    b = input('y_axis: ')
    test_case = np.array([a,b])
    test_case_tf = tf.constant(test_case, dtype= tf.float64)
    k_f = tf.constant(3)
    pred_scheme = Knn.knn(X_train_tf, Y_train_tf, X_test=test_case_tf, k=k_f)

    sess = tf.Session()
    top_y = sess.run(pred_scheme)

    print("Test data belong to cluster %d"%Knn.get_label(top_y))
    plt.clf()
    plt.title("Test data belong to cluster %d"%Knn.get_label(top_y))
    plt.plot( X1[:, 0], X1[:,1], 'ro', label='cluster 1')
    plt.plot(X2[:, 0], X2[:,1], 'bo', label='cluster 0')
    plt.plot(test_case[0], test_case[1], 'g', marker='D', markersize=10, label='test point')
    plt.legend(loc='best')
    plt.show()


