import gzip
import numpy as np
import matplotlib.pyplot as plt
import time
np.set_printoptions(threshold=np.inf)
from numba import cuda

#defining activation functions
def softmax(x):
    return np.exp(x-np.max(x)) / np.sum(np.exp(x-np.max(x)))
def ReLU(x):
    return np.maximum(np.zeros(x.shape), x)
def leaky_ReLU(x):
    return np.maximum(0.01*x, x)
def sigmoid(x):
    return 1/(1+np.exp(-x))
def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

#defining derivative of activation functions
def der_ReLU(x):
    if x >= 0:
        return 1
    else:
        return 0
def der_leaky_ReLU(x):
    if x >= 0:
        return 1
    else:
        return 0.01
def der_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))
def der_tanh(x):
    return (1-(np.power(tanh(x),2)))

class NeuralNetwork:

    def __init__(self, x1, y1, x2, y2, n):

        # base settings
        self.batch_num = 1
        self.train_data = x1
        self.train_label = y1
        self.test_data = x2
        self.test_label = y2
        self.max_iterations = n
        self.num_of_classes = (np.max(self.train_label)+1)
        self.num_of_input = len(self.train_data[0].flatten())

        # defining the onehot vectors
        self.onehot = np.zeros((self.num_of_classes, self.num_of_classes))
        for i in range(self.num_of_classes):
            self.onehot[i][i] = 1

        # settings for final connected layer
        self.final_layer_input = np.zeros(self.num_of_classes)
        self.final_layer_output = np.zeros(self.final_layer_input.shape)
        self.final_layer_bias = np.random.randint(-100, 100, size=self.num_of_classes) / 100
        self.final_layer_weights = np.random.randint(-100, 100, size=(self.num_of_classes, self.num_of_input)) / 100

        # partial derivative arrays
        self.d_loss_d_output = np.zeros(self.num_of_classes)
        self.d_loss_d_final_layer_weights = np.zeros(self.final_layer_weights.shape)
        self.d_loss_d_final_layer_bias = np.zeros(self.final_layer_bias.shape)

    def connected_layer(self, data4, weights, bias):
        input = np.zeros(bias.shape)
        for i in range(len(bias)):
            input[i] = np.dot(data4, weights[i]) + bias[i]
        return input

    def train(self):
        for iterations in range(self.max_iterations):
            for i in range(len(self.train_data)):

                # forward propogation
                self.final_layer_input = self.connected_layer(self.train_data[i].flatten(), self.final_layer_weights, self.final_layer_bias)
                self.final_layer_output = softmax(self.final_layer_input)

                # # debug
                # print(self.final_layer_output)
                # print(self.onehot[self.train_label[i]])
                # loss = -(np.log(self.final_layer_output[self.train_label[i]])+0.000000000001)
                # print(loss)

                # backward propogation
                for k in range(self.num_of_classes):
                    self.d_loss_d_output[k] += self.final_layer_output[k] - self.onehot[self.train_label[i]][k]

                if i%self.batch_num == 0:
                    for k in range(self.num_of_classes):
                        self.d_loss_d_final_layer_bias[k] += self.d_loss_d_output[k]
                        for l in range(self.num_of_input):
                            self.d_loss_d_final_layer_weights[k][l] += self.d_loss_d_output[k] * self.train_data[i].flatten()[l]

                    self.final_layer_weights -= 0.05 * self.d_loss_d_final_layer_weights
                    self.final_layer_bias -= 0.05 * self.d_loss_d_final_layer_bias

                    # reset partial derivatives
                    self.d_loss_d_output = np.zeros(self.num_of_classes)
                    self.d_loss_d_final_layer_weights = np.zeros(self.final_layer_weights.shape)
                    self.d_loss_d_final_layer_bias = np.zeros(self.final_layer_bias.shape)

                # for k in range(self.num_of_classes):
                #     self.d_loss_d_output[k] += self.final_layer_output[k] - self.onehot[self.train_label[i]][k]
                #     self.d_loss_d_final_layer_bias[k] += self.d_loss_d_output[k]
                #     for l in range(self.num_of_input):
                #         self.d_loss_d_final_layer_weights[k][l] += self.d_loss_d_output[k] * self.train_data[i].flatten()[l]

                # Updating the weights
                # self.final_layer_weights -= 0.05 * self.d_loss_d_final_layer_weights
                # self.final_layer_bias -= 0.05 * self.d_loss_d_final_layer_bias

                print(str(i + 1) + " images done!")
            print(str(iterations + 1) + " iterations done!")

    def evaluate_test(self):
        correct = 0
        for i in range(len(self.test_data)):
            self.final_layer_input = self.connected_layer(self.test_data[i].flatten(), self.final_layer_weights, self.final_layer_bias)
            self.final_layer_output = softmax(self.final_layer_input)
            if np.argmax(self.final_layer_output) == self.test_label[i]:
                correct += 1
        return (correct / len(self.test_data)) * 100

    def evaluate_train(self):
        correct = 0
        for i in range(len(self.train_data)):
            self.final_layer_input = self.connected_layer(self.train_data[i].flatten(), self.final_layer_weights, self.final_layer_bias)
            self.final_layer_output = softmax(self.final_layer_input)
            if np.argmax(self.final_layer_output) == self.train_label[i]:
                correct += 1
        # print(self.first_layer_filter)
        return (correct / len(self.train_data)) * 100

# Read the train and test data and labels
f = gzip.open('train-images-idx3-ubyte.gz','r')
f.read(16)
train_instance = np.frombuffer(f.read(60000*28*28), dtype=np.uint8).astype(np.float32)
train_instance = train_instance.reshape(60000,28,28,1)
train_instance = (train_instance/255)
f.close()
f = gzip.open('train-labels-idx1-ubyte.gz','r')
f.read(8)
train_instance_label = np.frombuffer(f.read(60000), dtype=np.uint8).astype(np.int64)
f.close()
f = gzip.open('t10k-images-idx3-ubyte.gz','r')
f.read(16)
test_instance = np.frombuffer(f.read(10000*28*28), dtype=np.uint8).astype(np.float32)
test_instance = test_instance.reshape(10000,28,28,1)
test_instance = (test_instance/255)
f.close()
f = gzip.open('t10k-labels-idx1-ubyte.gz','r')
f.read(8)
test_instance_label = np.frombuffer(f.read(10000), dtype=np.uint8).astype(np.int64)
f.close()


NN = NeuralNetwork(train_instance, train_instance_label, test_instance, test_instance_label, 50)
train_NN.train()
x = train_NN.evaluate_train()
y = train_NN.evaluate_test()
print(x,y)

