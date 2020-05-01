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

# define the neural network class and functions
'''
class NeuralNetwork:

    def __init__(self, train_data, train_label):
        self.train_data = train_data
        self.train_label = train_label
        self.num_of_classes = np.max(self.train_label) + 1
        self.num_of_input = len(self.train_data[0].flatten())
        self.bias = np.zeros(self.num_of_classes)
        self.weights = np.zeros((self.num_of_classes, self.num_of_input))
        self.output = np.zeros(self.num_of_classes)
        self.onehot = np.zeros((self.num_of_classes, self.num_of_classes))
        for i in range(self.num_of_classes):
            self.onehot[i][i] = 1
        print("Number of classes: "+str(self.num_of_classes))
        print("Number of inputs per data: "+str(self.num_of_input))
        print("Number of instances of data: " + str(len(self.train_data)))
        time.sleep(1)

    # @cuda.jit
    def train(self, max_iterations):
        self.max_iterations = max_iterations
        for iterations in range(self.max_iterations):
            for i in range(len(self.train_data)):
                for j in range(len(self.output)):
                    self.output[j] = np.dot(self.weights[j],self.train_data[i].flatten()) + self.bias[j]
                self.output = softmax(self.output)
                # print(np.sum(self.output))
                loss = -(np.log(self.output[train_label[i]])+0.00000000000001)
                self.d_loss_d_output = np.zeros(self.num_of_classes)
                self.d_loss_d_w = np.zeros(self.weights.shape)
                self.d_loss_d_bias = np.zeros(self.num_of_classes)
                for k in range(self.num_of_classes):
                    self.d_loss_d_output[k] = self.output[k] - self.onehot[self.train_label[i]][k]
                    self.d_loss_d_bias[k] = self.output[k] - self.onehot[self.train_label[i]][k]
                    for l in range(self.num_of_input):
                        self.d_loss_d_w[k][l] = 0.5 * self.d_loss_d_output[k] * self.train_data[i].flatten()[l]
                self.weights -= self.d_loss_d_w
                self.bias -= self.d_loss_d_bias
                print(loss)
                # print(self.output)
                print(str(i)+" images done!")
            print(str(iterations+1) + " iterations Done!")
            time.sleep(1)

    def evaluate(self, test_data, test_label):
        correct = 0
        for i in range(len(test_data)):
            for j in range(len(self.output)):
                self.output[j] = np.dot(self.weights[j],train_data[i].flatten()) + self.bias[j]
            self.output = softmax(self.output)
            if np.argmax(self.output) == test_label[i]:
                correct += 1
        print("The accuracy of the predictions is " + str((correct/len(test_data))*100) + "%.")

    def predict(self, test_data, test_label):
        for m in range(len(self.output)):
            self.output[m] = np.dot(self.weights[m],train_data.flatten()) + self.bias[j]
        self.output = softmax(self.output)
        print("The data is predicted to be in class " + str(np.argmax(self.output))+".")

'''

class NeuralNetwork:

    def __init__(self, x1, y1, x2, y2, n):
        self.train_data = x1
        self.train_label = y1
        self.test_data = x2
        self.test_label = y2
        self.max_iterations = n
        self.num_of_classes = (np.max(self.train_label)+1)
        self.num_of_input = len(self.train_data[0].flatten())
        self.bias = np.zeros(self.num_of_classes)
        self.weights = np.zeros((self.num_of_classes, self.num_of_input))
        self.output = np.zeros(self.num_of_classes)
        self.onehot = np.zeros((self.num_of_classes,self.num_of_classes))
        self.d_loss_d_output = np.zeros(self.num_of_classes)
        self.d_loss_d_w = np.zeros(self.weights.shape)
        self.d_loss_d_bias = np.zeros(self.num_of_classes)
        for i in range(self.num_of_classes):
            self.onehot[i][i] = 1

    def train(self):
        for iterations in range(self.max_iterations):
            for i in range(len(self.train_data)):
                for j in range(len(self.output)):
                    self.output[j] = np.dot(self.weights[j],self.train_data[i].flatten()) + self.bias[j]
                self.output = softmax(self.output)
                # print("Input")
                # print(self.train_data[i])
                # print("Output")
                # print(self.output)
                # print("Actual Output")
                # print(self.onehot[self.train_label[i]])
                # loss = -(np.log(self.output[self.train_label[i]])+0.000000000001)
                # print("Loss")
                # print(loss)
                for k in range(self.num_of_classes):
                    self.d_loss_d_output[k] = self.output[k] - self.onehot[self.train_label[i]][k]
                    self.d_loss_d_bias[k] = 0.05*self.output[k] - self.onehot[self.train_label[i]][k]
                    for l in range(self.num_of_input):
                        self.d_loss_d_w[k][l] = 0.05*self.d_loss_d_output[k] * self.train_data[i].flatten()[l]
                # print("Weights before update")
                # print(self.weights)
                # print("Diff of weights")
                # print(self.d_loss_d_w)
                self.weights -= self.d_loss_d_w
                self.bias -= self.d_loss_d_bias
                print(str(i+1) + " images done!")
                # print("Weights after update")
                # print(self.weights)
                # time.sleep(3)
            print(str(iterations+1) + " images done!")

    def evaluate_test(self):
        correct = 0
        for i in range(len(self.test_data)):
            for j in range(len(self.output)):
                self.output[j] = np.dot(self.weights[j],self.test_data[i].flatten()) + self.bias[j]
            self.output = softmax(self.output)
            if np.argmax(self.output) == self.test_label[i]:
                correct += 1
        # print("The accuracy of the predictions is " + str((correct/len(self.test_data))*100) + "%.")
        return (correct/len(self.test_data))*100

    def evaluate_train(self):
        correct = 0
        for i in range(len(self.train_data)):
            for j in range(len(self.output)):
                self.output[j] = np.dot(self.weights[j],self.train_data[i].flatten()) + self.bias[j]
            self.output = softmax(self.output)
            if np.argmax(self.output) == self.train_label[i]:
                correct += 1
        # print("The accuracy of the predictions is " + str((correct/len(self.test_data))*100) + "%.")
        return (correct/len(self.train_data))*100

print(cuda.gpus)
# time.sleep(2)
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

print("x: " + str(x))
print("y: " + str(y))
# print("z: " + str(z))