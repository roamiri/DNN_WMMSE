
import time
# import network3
import network2
# from network3 import sigmoid, tanh, ReLU, Network
# from network3 import FullyConnectedLayer
# import function_wmmse_powercontrol as wmmse
import pickle

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

training_data = []
for i in range(0, Xtrain.shape[1]):
    training_data.append((Xtrain[:, i].reshape(100, 1), Ytrain[:, i].reshape(10, 1)))


test_data = []
for i in range(0, Xtest.shape[1]):
    test_data.append((Xtest[:, i].reshape(100, 1), Ytest[:, i].reshape(10, 1)))


mini_batch_size = 1000
training_epochs = 10
eta=0.1
lmbda = 10.0

net = network2.Network([K**2, 200, 200, 200, K], cost=network2.QuadraticCost)

net.SGD(training_data, training_epochs, mini_batch_size, eta, lmbda, test_data,
        monitor_evaluation_cost=True,
        monitor_evaluation_accuracy=True,
        monitor_training_cost=True,
        monitor_training_accuracy=True
        )
