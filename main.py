
import time
import network
import pickle
from sklearn import preprocessing
import matplotlib.pyplot as plt

K = 10  # Number of users

data_file = open("./data/Xtrain.pickle","rb")
Xtrain = pickle.load(data_file)
data_file.close()

data_file = open("./data/Ytrain.pickle","rb")
Ytrain = pickle.load(data_file)
data_file.close()

data_file = open("./data/Xeval.pickle","rb")
Xeval = pickle.load(data_file)
data_file.close()

data_file = open("./data/Yeval.pickle","rb")
Yeval = pickle.load(data_file)
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
for i in range(0, Xtrain.shape[1]/5):
    training_data.append((scaler.fit_transform(Xtrain[:, i].reshape(100, 1)), Ytrain[:, i].reshape(10, 1)))

for i in range(0, Yeval.shape[1]):
    Yeval[:, i][Yeval[:, i] < 0.5] = 0.0
    Yeval[:, i][Yeval[:, i] >= 0.5] = 1.0

eval_data = []
for i in range(0, Xeval.shape[1]/5):
    eval_data.append((scaler.fit_transform(Xeval[:, i].reshape(100, 1)), Yeval[:, i].reshape(10, 1)))

for i in range(0, Ytest.shape[1]):
    Ytest[:, i][Ytest[:, i] < 0.5] = 0.0
    Ytest[:, i][Ytest[:, i] >= 0.5] = 1.0

test_data = []
for i in range(0, Xtest.shape[1]/5):
    test_data.append((scaler.fit_transform(Xtest[:, i].reshape(100, 1)), Ytest[:, i].reshape(10, 1)))

data_size = len(training_data)
mini_batch_size = 10
training_epochs = 90
eta = 5.0
lmbda = 0.0

DNN = network.Network([K**2, 200, 100, K], cost=network.QuadraticCost)

# evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = []
T = True
(evaluation_cost, evaluation_accuracy, training_cost, training_accuracy) = DNN.SGD(training_data, training_epochs, mini_batch_size, eta, lmbda, eval_data,
        monitor_evaluation_cost=T,
        monitor_evaluation_accuracy=T,
        monitor_training_cost=T,
        monitor_training_accuracy=T
        )

save = False
if save:
    # with open('./data/three_{0}_{1}_{2}_{3}_{4}.pkl'.format('200,200,200', eta, lmbda, mini_batch_size, data_size), 'wb') as f:
    with open('./data/Quadratic.pkl','wb') as f:
        pickle.dump(training_accuracy, f)
        pickle.dump(training_cost, f)
        pickle.dump(evaluation_accuracy, f)
        pickle.dump(evaluation_cost, f)

load = False
if load:
    with open('./data/three_{0}_{1}_{2}_{3}_{4}.pkl'.format(200, eta, lmbda, mini_batch_size, data_size), 'rb') as f:
        training_accuracy = pickle.load(f)
        training_cost = pickle.load(f)
        evaluation_accuracy = pickle.load(f)
        evaluation_cost = pickle.load(f)

plotOutput = False
if plotOutput:
    plt.plot(training_accuracy)
    plt.plot(evaluation_accuracy)
    plt.ylabel('Training accuracy')
    plt.xlabel('Epochs')
    plt.grid(True, color='k', linestyle=':', linewidth=0.3)
    axes = plt.gca()
    # ymax = max(training_accuracy)
    # xpos = training_accuracy.index(ymax)
    # axes.annotate('max={}%'.format(ymax), xy=(xpos, ymax), xytext=(xpos-10, ymax - 10),arrowprops=dict(facecolor='black', shrink=0.05))
    # axes.set_xlim([xmin, xmax])
    axes.set_ylim([0.0, 100.0])
    axes.legend(['Training data', 'Test data'], loc='upper left')
    plt.show()
