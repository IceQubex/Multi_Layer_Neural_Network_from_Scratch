import gzip
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import random
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

        # settings for first connected layer
        self.first_layer_num = 10
        self.first_layer_input = np.zeros(self.first_layer_num)
        self.first_layer_output = np.zeros(self.first_layer_input.shape)
        self.first_layer_bias = np.random.randint(-100, 100, size=self.first_layer_num) / 100
        self.first_layer_weights = np.random.randint(-100, 100, size=(self.first_layer_num, self.num_of_input))/ 100

        # settings for final connected layer
        self.final_layer_input = np.zeros(self.num_of_classes)
        self.final_layer_output = np.zeros(self.final_layer_input.shape)
        self.final_layer_bias = np.random.randint(-100, 100, size=self.num_of_classes) / 100
        self.final_layer_weights = np.random.randint(-100, 100, size=(self.num_of_classes, self.first_layer_num)) / 100

    def connected_layer(self, data4, weights, bias):
        input = np.zeros(bias.shape)
        for i in range(len(bias)):
            input[i] = np.dot(data4, weights[i]) + bias[i]
        return input

    def train(self):
        for iterations in range(self.max_iterations):
            for i in range(len(self.train_data)):

                # settings for derivatives of first layer
                self.d_loss_d_first_layer_weights = np.zeros(self.first_layer_weights.shape)
                self.d_loss_d_first_layer_bias = np.zeros(self.first_layer_bias.shape)

                # settings for derivatives of final layer
                self.d_loss_d_output = np.zeros(self.num_of_classes)
                self.d_loss_d_final_layer_weights = np.zeros(self.final_layer_weights.shape)
                self.d_loss_d_final_layer_bias = np.zeros(self.final_layer_bias.shape)

                # forward propogation
                self.first_layer_input = self.connected_layer(self.train_data[i].flatten(), self.first_layer_weights, self.first_layer_bias)
                self.first_layer_output = sigmoid(self.first_layer_input)
                self.final_layer_input = self.connected_layer(self.first_layer_output, self.final_layer_weights, self.final_layer_bias)
                self.final_layer_output = softmax(self.final_layer_input)

                # calculate the loss
                loss = -(np.log(self.final_layer_output[self.train_label[i]])+0.000000000001) # to prevent calculating the log of 0, which would give infinity

                # backward propagation
                for k in range(self.num_of_classes):
                    self.d_loss_d_output[k] += self.final_layer_output[k] - self.onehot[self.train_label[i]][k]
                    self.d_loss_d_final_layer_bias[k] += self.d_loss_d_output[k]
                    for l in range(self.first_layer_num):
                        self.d_loss_d_final_layer_weights[k][l] += self.d_loss_d_output[k] * self.first_layer_output[l]
                        self.d_loss_d_first_layer_bias[l] += self.d_loss_d_output[k] * self.final_layer_weights[k][l] * der_sigmoid(self.first_layer_input[l])
                        for m in range(self.num_of_input):
                            self.d_loss_d_first_layer_weights[l][m] += self.d_loss_d_output[k] * self.final_layer_weights[k][l] * der_sigmoid(self.first_layer_input[l]) * self.train_data[i].flatten()[m]

                # Updating the weights
                self.first_layer_weights -= 0.05 * self.d_loss_d_first_layer_weights
                self.first_layer_bias -= 0.05 * self.d_loss_d_first_layer_bias
                self.final_layer_weights -= 0.05 * self.d_loss_d_final_layer_weights
                self.final_layer_bias -= 0.05 * self.d_loss_d_final_layer_bias

                print("Iteration number " + str(iterations+1) + ": " + str(i + 1) + " images done!")
            # print(str(iterations + 1) + " iterations done!")

    def evaluate_test(self):
        correct = 0
        for i in range(len(self.test_data)):
            self.first_layer_input = self.connected_layer(self.test_data[i].flatten(), self.first_layer_weights, self.first_layer_bias)
            self.first_layer_output = sigmoid(self.first_layer_input)
            self.final_layer_input = self.connected_layer(self.first_layer_output, self.final_layer_weights, self.final_layer_bias)
            self.final_layer_output = softmax(self.final_layer_input)
            if np.argmax(self.final_layer_output) == self.test_label[i]:
                correct += 1
        return (correct / len(self.test_data)) * 100

    def evaluate_train(self):
        correct = 0
        for i in range(len(self.train_data)):
            self.first_layer_input = self.connected_layer(self.train_data[i].flatten(), self.first_layer_weights, self.first_layer_bias)
            self.first_layer_output = sigmoid(self.first_layer_input)
            self.final_layer_input = self.connected_layer(self.first_layer_output, self.final_layer_weights, self.final_layer_bias)
            self.final_layer_output = softmax(self.final_layer_input)
            if np.argmax(self.final_layer_output) == self.train_label[i]:
                correct += 1
        return (correct / len(self.train_data)) * 100


# Read the train and test data and labels
f = gzip.open('train-images-idx3-ubyte.gz','r')
f.read(16)
total_train = np.frombuffer(f.read(60000*28*28), dtype=np.uint8).astype(np.float32)
total_train = total_train.reshape(60000,28,28,1)
train_instance, validation_instance = np.split(total_train, [50000])
train_instance = (train_instance/255)
validation_instance = (validation_instance/255)
f.close()
f = gzip.open('train-labels-idx1-ubyte.gz','r')
f.read(8)
total_train_label = np.frombuffer(f.read(60000), dtype=np.uint8).astype(np.int64)
train_instance_label, validation_instance_label = np.split(total_train_label, [50000])
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

NN = NeuralNetwork(train_instance, train_instance_label, validation_instance, validation_instance_label, 1)
NN.train()
x = NN.evaluate_train()
y = NN.evaluate_test()
print("The performance of the neural network on the training set is " + str(x) + "%.")
print("The performance of the neural network on the test set is " + str(y) + "%.")


'''

# XOR Data-set
data = [[-10,10],[-9,9],[-8,8],[-7,7],[-6,6],[-5,5],[-4,4],[-3,3],[-2,2],[-1,1],[1,-1],[2,-2],[3,-3],[4,-4],[5,-5],[6,-6],[7,-7],[8,-8],[9,-9],[10,-10],[-10,-10],[-9,-9],[-8,-8],[-7,-7],[-6,-6],[-5,-5],[-4,-4],[-3,-3],[-2,-2],[-1,-1],[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9],[10,10]]
label = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

temp = list(zip(data, label))
random.shuffle(temp)
length1 = len(temp)
# train_data, train_label = zip(*temp)
temp_train = temp[:math.floor(0.75*length1)]
temp_test = temp[math.floor(0.75*length1):]
train_x, train_y = zip(*temp_train)
test_x, test_y = zip(*temp_test)
train_x = np.array(train_x)
train_y = np.array(train_y)
test_x = np.array(test_x)
test_y = np.array(test_y)

test = NeuralNetwork(train_x, train_y, test_x, test_y, 1000)
test.train()
print("Done")
print(test.evaluate_train(), test.evaluate_test())
'''