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
    return np.maximum(0.01*x,x)
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
        return 0.001
def der_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))
def der_tanh(x):
    return (1-(np.power(tanh(x),2)))

class NeuralNetwork:

    def __init__(self, x1, y1, x2, y2, n):

        #base settings
        self.train_data = x1
        self.train_label = y1
        self.test_data = x2
        self.test_label = y2
        self.max_iterations = n
        self.num_of_classes = (np.max(self.train_label)+1)
        self.num_of_input = len(self.train_data[0].flatten())

        #defining the onehot vectors
        self.onehot = np.zeros((self.num_of_classes,self.num_of_classes))
        for i in range(self.num_of_classes):
            self.onehot[i][i] = 1

        #settings for first layer
        self.first_layer_num = 20
        self.first_layer_input = np.zeros(self.first_layer_num)
        self.first_layer_output = np.zeros(self.first_layer_input.shape)
        self.first_layer_bias = np.random.randint(-10000, 10000, size=self.first_layer_num)
        self.first_layer_bias = self.first_layer_bias/10000
        self.first_layer_weights = np.random.randint(-10000, 10000, size=(self.first_layer_num, self.num_of_input))
        self.first_layer_weights = self.first_layer_weights/10000

        # settings for final layer
        self.final_layer_input = np.zeros(self.num_of_classes)
        self.final_layer_output = np.zeros(self.final_layer_input.shape)
        self.final_layer_bias = np.random.randint(-10000, 10000, size=self.num_of_classes)
        self.final_layer_bias = self.final_layer_bias/10000
        self.final_layer_weights = np.random.randint(-10000, 10000, size=(self.num_of_classes, self.first_layer_num))
        self.final_layer_weights = self.final_layer_weights/10000

    def train(self):
        # count = 0
        for iterations in range(self.max_iterations):
            for i in range(len(self.train_data)):

                # settings for derivatives of first layer
                self.d_loss_d_first_layer_weights = np.zeros(self.first_layer_weights.shape)
                self.d_loss_d_first_layer_bias = np.zeros(self.first_layer_bias.shape)

                # settings for derivatives of final layer
                self.d_loss_d_output = np.zeros(self.num_of_classes)
                self.d_loss_d_final_layer_weights = np.zeros(self.final_layer_weights.shape)
                self.d_loss_d_final_layer_bias = np.zeros(self.final_layer_bias.shape)

                for j in range(self.first_layer_num):
                    self.first_layer_input[j] = np.dot(self.first_layer_weights[j],self.train_data[i].flatten()) + self.first_layer_bias[j]
                self.first_layer_output = sigmoid(self.first_layer_input)
                for j in range(self.num_of_classes):
                    self.final_layer_input[j] = np.dot(self.final_layer_weights[j],self.first_layer_output) + self.final_layer_bias[j]
                self.final_layer_output = softmax(self.final_layer_input)
                # print(np.sum(self.final_layer_output))
                # print(self.final_layer_output)
                # print(self.onehot[self.train_label[i]])
                loss = -(np.log(self.final_layer_output[self.train_label[i]])+0.000000000001)
                # print(loss)
                # time.sleep(0.5)
                for k in range(self.num_of_classes):
                    self.d_loss_d_output[k] += self.final_layer_output[k] - self.onehot[self.train_label[i]][k]
                    self.d_loss_d_final_layer_bias[k] += self.d_loss_d_output[k]
                    for l in range(self.first_layer_num):
                        self.d_loss_d_final_layer_weights[k][l] += self.d_loss_d_output[k] * self.first_layer_output[l]
                        self.d_loss_d_first_layer_bias[l] += self.d_loss_d_output[k]*self.final_layer_weights[k][l]*der_sigmoid(self.first_layer_input[l])
                        for m in range(self.num_of_input):
                            self.d_loss_d_first_layer_weights[l][m] += self.d_loss_d_output[k]*self.final_layer_weights[k][l]*der_sigmoid(self.first_layer_input[l])*self.train_data[i].flatten()[m]
                # print("Output")
                # print(self.d_loss_d_output)
                # print("First layer weights")
                # print(self.d_loss_d_first_layer_weights)
                # print("First layer bias")
                # print(self.d_loss_d_first_layer_bias)
                # print("Second layer weights")
                # print(self.d_loss_d_final_layer_weights)
                # print("Second layer bias")
                # print(self.d_loss_d_final_layer_bias)
                # time.sleep(2)
                # self.first_layer_weights -= np.exp(-(0.0001*count-0.5))*self.d_loss_d_first_layer_weights
                # self.first_layer_bias -= np.exp(-(0.0001*count-0.5))*self.d_loss_d_first_layer_bias
                # self.final_layer_weights -= np.exp(-(0.0001*count-0.5))*self.d_loss_d_final_layer_weights
                # self.final_layer_bias -= np.exp(-(0.0001*count-0.5))*self.d_loss_d_final_layer_bias
                self.first_layer_weights -= 1*self.d_loss_d_first_layer_weights
                self.first_layer_bias -= 1*self.d_loss_d_first_layer_bias
                self.final_layer_weights -= 1*self.d_loss_d_final_layer_weights
                self.final_layer_bias -= 1*self.d_loss_d_final_layer_bias
                print(str(i+1) + " images done!")
                # count+=1
            print(str(iterations+1) + " iterations done!")

    def evaluate_test(self):
        correct = 0
        for i in range(len(self.test_data)):
            for j in range(self.first_layer_num):
                self.first_layer_input[j] = np.dot(self.first_layer_weights[j], self.test_data[i].flatten()) + self.first_layer_bias[j]
            self.first_layer_output = sigmoid(self.first_layer_input)
            for j in range(self.num_of_classes):
                self.final_layer_input[j] = np.dot(self.final_layer_weights[j], self.first_layer_output) + self.final_layer_bias[j]
            self.final_layer_output = softmax(self.final_layer_input)
            if np.argmax(self.final_layer_output) == self.test_label[i]:
                correct += 1
        # print("The accuracy of the predictions is " + str((correct/len(self.test_data))*100) + "%.")
        return (correct/len(self.test_data))*100

    def evaluate_train(self):
        correct = 0
        for i in range(len(self.train_data)):
            for j in range(self.first_layer_num):
                self.first_layer_input[j] = np.dot(self.first_layer_weights[j], self.train_data[i].flatten()) + self.first_layer_bias[j]
            self.first_layer_output = sigmoid(self.first_layer_input)
            for j in range(self.num_of_classes):
                self.final_layer_input[j] = np.dot(self.final_layer_weights[j], self.first_layer_output) + self.final_layer_bias[j]
            self.final_layer_output = softmax(self.final_layer_input)
            if np.argmax(self.final_layer_output) == self.train_label[i]:
                correct += 1
        # print("The accuracy of the predictions is " + str((correct/len(self.test_data))*100) + "%.")
        return (correct/len(self.train_data))*100

# print(cuda.gpus)

# Read the train and test data and labels
f = gzip.open('train-images-idx3-ubyte.gz','r')
f.read(16)
train_instance = np.frombuffer(f.read(60000*28*28), dtype=np.uint8).astype(np.float32)
train_instance = train_instance.reshape(60000,28,28,1)
train_instance = (train_instance/25500)
f.close()
f = gzip.open('train-labels-idx1-ubyte.gz','r')
f.read(8)
train_instance_label = np.frombuffer(f.read(60000), dtype=np.uint8).astype(np.int64)
f.close()
f = gzip.open('t10k-images-idx3-ubyte.gz','r')
f.read(16)
test_instance = np.frombuffer(f.read(10000*28*28), dtype=np.uint8).astype(np.float32)
test_instance = test_instance.reshape(10000,28,28,1)
test_instance = (test_instance/25500)
f.close()
f = gzip.open('t10k-labels-idx1-ubyte.gz','r')
f.read(8)
test_instance_label = np.frombuffer(f.read(10000), dtype=np.uint8).astype(np.int64)
f.close()


train_NN = NeuralNetwork(train_instance, train_instance_label, test_instance, test_instance_label, 1)
train_NN.train()
x = train_NN.evaluate_train()
y = train_NN.evaluate_test()

# test_NN = NeuralNetwork(test_instance, test_instance_label, test_instance, test_instance_label, 1)
# test_NN.train()
# y = test_NN.evaluate()

# proper_NN = NeuralNetwork(train_instance, train_instance_label, test_instance, test_instance_label, 1)
# proper_NN.train()
# z = proper_NN.evaluate()

print("The accuracy of the neural network on the training set is  " + str(x) + "%.")
print("The accuracy of the neural network on the test set is " + str(y) + "%.")