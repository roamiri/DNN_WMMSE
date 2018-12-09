
import time
import network
import pickle
from sklearn import preprocessing

K = 10  # Number of users

data_file = open("./data/Xtrain.pickle","rb")
Xtrain = pickle.load(data_file)
data_file.close()

data_file = open("./data/Ytrain.pickle","rb")
Ytrain = pickle.load(data_file)
data_file.close()

data_file = open("./data/Xtest.pickle","rb")
Xtest = pickle.load(data_file)
data_file.close()

data_file = open("./data/Ytest.pickle","rb")
Ytest = pickle.load(data_file)
data_file.close()

scaler = preprocessing.MinMaxScaler()

for i in range(0, Ytrain.shape[1]):
    Ytrain[:, i][Ytrain[:, i] < 0.5] = 0.0
    Ytrain[:, i][Ytrain[:, i] >= 0.5] = 1.0

training_data = []
for i in range(0, Xtrain.shape[1]):
    training_data.append((scaler.fit_transform(Xtrain[:, i].reshape(100, 1)), Ytrain[:, i].reshape(10, 1)))


for i in range(0, Ytest.shape[1]):
    Ytest[:, i][Ytest[:, i] < 0.5] = 0.0
    Ytest[:, i][Ytest[:, i] >= 0.5] = 1.0

test_data = []
for i in range(0, Xtest.shape[1]):
    test_data.append((scaler.fit_transform(Xtest[:, i].reshape(100, 1)), Ytest[:, i].reshape(10, 1)))

mini_batch_size = 10
training_epochs = 100
eta = 3.0
lmbda = 0.0

DNN = network.Network([K**2, 200, 200, K], cost=network.QuadraticCost)

# evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = []

(evaluation_cost, evaluation_accuracy, training_cost, training_accuracy) = DNN.SGD(training_data, training_epochs, mini_batch_size, eta, lmbda, test_data,
        monitor_evaluation_cost=False,
        monitor_evaluation_accuracy=False,
        monitor_training_cost=False,
        monitor_training_accuracy=False
        )

save = True
if save:
    with open('./data/three_{0}_{1}_{2}_{3}.pkl'.format('200,200', eta, lmbda, mini_batch_size), 'wb') as f:
        pickle.dump(training_accuracy, f)
        pickle.dump(training_cost, f)
        pickle.dump(evaluation_accuracy, f)
        pickle.dump(evaluation_cost, f)

load = False
if load:
    with open('./data/three_{0}_{1}_{2}_{3}.pkl'.format(200, eta, lmbda, mini_batch_size), 'rb') as f:
        training_accuracy = pickle.load(f)
        training_cost = pickle.load(f)
        evaluation_accuracy = pickle.load(f)
        evaluation_cost = pickle.load(f)

import matplotlib.pyplot as plt

plotOutput = True
if plotOutput:
    plt.plot(training_accuracy)
    plt.ylabel('Training accuracy')
    plt.xlabel('Epochs')
    plt.grid
    plt.show()