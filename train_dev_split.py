import glob
import os

from sklearn.model_selection import train_test_split


def one_random_split(dev_size):

    i_l_ROOT = 'images/'
    f = os.listdir('images')

    X = []
    Y = []

    for folder in f:
        direct = i_l_ROOT + folder + '/'
        l_direct = direct + folder + ".txt"


        with open(l_direct) as label_file:
            label_list = label_file.read().splitlines()

        file_list = glob.glob(direct + '*.jpg')
        for i in range(len(file_list)):
            X.append(file_list[i])
            Y.append(label_list[i])

    X_train, X_dev, Y_train, Y_dev = train_test_split(X, Y, test_size=dev_size)


    return X_train, X_dev, Y_train, Y_dev


def k_split_sample(k_set, size):
    db_ROOT = 'db/'

    for k in range(k_set):

        db_direct = db_ROOT + "db" + str(k) + "/"
        if not os.path.exists(db_direct):
            os.makedirs(db_direct)
        else:
            trash_files = os.listdir(db_direct)
            for trash in trash_files:
                os.remove(db_direct + trash)

        train_file = open(db_direct + "train.txt", "w+")
        val_file = open(db_direct + "val.txt", "w+")
        dev_file = open(db_direct + "dev.txt", "w+")
        lbtrain_file = open(db_direct + "lbtrain.txt", "w+")
        lbval_file = open(db_direct + "lbval.txt", "w+")
        lbdev_file = open(db_direct + "lbdev.txt", "w+")

        X_train, X_dev, Y_train, Y_dev = one_random_split(dev_size= size)

        for i in range(len(X_train)):
             train_file.write(X_train[i] + "\n")
             lbtrain_file.write(Y_train[i] + "\n")

        for i in range(len(X_dev)):
             dev_file.write(X_dev[i] + "\n")
             lbdev_file.write(Y_dev[i] + "\n")

        train_file.close()
        val_file.close()
        dev_file.close()
        lbtrain_file.close()
        lbval_file.close()
        lbdev_file.close()
#k_split_sample(5,0.2)