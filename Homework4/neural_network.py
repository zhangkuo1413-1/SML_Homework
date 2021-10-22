import numpy as np
from matplotlib import pyplot as plt

def read_data():
    #read all the training data and test data
    train_in = np.loadtxt("dataSets/mnist_small_train_in.txt", delimiter=",")
    train_out = np.loadtxt("dataSets/mnist_small_train_out.txt")
    test_in = np.loadtxt("dataSets/mnist_small_test_in.txt", delimiter=",")
    test_out = np.loadtxt("dataSets/mnist_small_test_out.txt")
    return train_in, train_out, test_in, test_out

def make_onehot(x):
    #transform the label into one-hot vector
    output = np.zeros([len(x), 10])
    for i in range(len(x)):
        output[i, int(x[i])] = 1
    return output

#activation function
def relu(x):
    return x * (x > 0)

def d_relu(x):
    return 1.0 * (x != 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    return x * (1 - x)

class NeuralNetwork():
    #the class of neural network
    def __init__(self, num_input, num_h1, num_h2, num_output, acti, d_acti, two_hidden=True):
        self.acti = acti
        self.d_acti = d_acti
        self.two_hidden = two_hidden
        self.W1 = np.random.randn(num_h1, num_input) * 0.1
        self.b1 = np.zeros(num_h1)
        if two_hidden:
            self.W2 = np.random.randn(num_h2, num_h1) * 0.1
            self.b2 = np.zeros(num_h2)
        else:
            num_h2=num_h1
        self.W3 = np.random.randn(num_output, num_h2) * 0.1
        self.b3 = np.zeros(num_output)

    def softmax(self, x):
        return np.exp(x) / np.exp(x).sum(axis=1).reshape(-1, 1)

    def cross_entropy(self, output, label):
        return (-np.log(output) * label).sum()

    def d_loss(self, output, label):
        return (output - label)

    def forward(self, x):
        self.input = x
        #first hidden layer
        affi1 = np.dot(self.W1, self.input.T) + self.b1[:, np.newaxis]
        self.h1 = self.acti(affi1)
        #second hidden layer
        if self.two_hidden:
            affi2 = np.dot(self.W2, self.h1) + self.b2[:, np.newaxis]
            self.h2 = self.acti(affi2)
        else:
            self.h2 = self.h1
        #output layer
        affi3 = (np.dot(self.W3, self.h2) + self.b3[:, np.newaxis]).T
        self.output = self.softmax(affi3)
        return self.output

    def backward(self, y_true):
        delta = self.d_loss(self.output, y_true)
        self.dW3 = np.dot(delta.T, self.h2.T)
        self.db3 = np.sum(delta, axis=0)
        delta = np.dot(self.W3.T, delta.T) * self.d_acti(self.h2)
        if self.two_hidden:
            self.dW2 = np.dot(delta, self.h1.T)
            self.db2 = np.sum(delta, axis=1)
            delta = np.dot(self.W2.T, delta) * self.d_acti(self.h1)
        self.dW1 = np.dot(delta, self.input)
        self.db1 = np.sum(delta, axis=1)

    def update(self, learning_rate):
        self.W1 -= learning_rate * self.dW1
        self.b1 -= learning_rate * self.db1
        if self.two_hidden:
            self.W2 -= learning_rate * self.dW2
            self.b2 -= learning_rate * self.db2
        self.W3 -= learning_rate * self.dW3
        self.b3 -= learning_rate * self.db3

def train_steps(model, learning_rate, n_epochs, batch_size, train_in, train_out, test_in, test_out): 
    #function to train the model
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    for i in range(n_epochs):
        samples = np.arange(train_in.shape[0])
        batch = np.random.choice(samples, size=batch_size, replace=False)
        loss1 = 0
        acc1 = 0
        loss2 = 0
        acc2 = 0
        data = train_in[batch, :]
        labels = train_out[batch]
        y_true = make_onehot(labels)
        y_pred = model.forward(data)
        loss1 = (model.cross_entropy(y_pred, y_true) / batch_size)
        acc1 = ((labels==y_pred.argmax(axis=1)).sum() / batch_size)
        model.backward(y_true)
        model.update(learning_rate)
        for j in range(test_in.shape[0]):
            data = test_in[j: j+1, :]
            labels = test_out[j: j+1]
            y_true = make_onehot(labels)
            y_pred = model.forward(data)
            loss2 += model.cross_entropy(y_pred, y_true)
            acc2 += (labels==y_pred.argmax(axis=1)).sum()
        train_loss.append(loss1)
        test_loss.append(loss2/test_in.shape[0])
        train_acc.append(acc1)
        test_acc.append(acc2/test_in.shape[0])
        #print("Epoch: {}, train loss: {:.6f}, train accuracy:{:.6f}, test loss: {:.6f}, test accuracy: {:.6f}".format(i, loss1, acc1, loss2/test_in.shape[0], acc2/test_in.shape[0]))
    return train_loss, test_loss, train_acc, test_acc

#compare of models with different structure
train_in, train_out, test_in, test_out = read_data()
dim = train_in.shape[1]
model1 = NeuralNetwork(dim, 1024, None, 10, relu, d_relu, two_hidden=False)
result1 = train_steps(model1, 0.001, 500, 256, train_in, train_out, test_in, test_out)
model2 = NeuralNetwork(dim, 256, None, 10, relu, d_relu, two_hidden=False)
result2 = train_steps(model2, 0.001, 500, 256, train_in, train_out, test_in, test_out)
model3 = NeuralNetwork(dim, 512, 128, 10, relu, d_relu)
result3 = train_steps(model3, 0.001, 500, 256, train_in, train_out, test_in, test_out)
model4 = NeuralNetwork(dim, 128, 32, 10, relu, d_relu)
result4 = train_steps(model4, 0.001, 500, 256, train_in, train_out, test_in, test_out)
plt.figure()
plt.plot(np.arange(500), result1[3], linewidth = 0.8, label="1 hidden layer with 1024 elements")
plt.plot(np.arange(500), result2[3], linewidth = 0.8, label="1 hidden layer with 256 elements")
plt.plot(np.arange(500), result3[3], linewidth = 0.8, label="2 hidden layer with 512 and 128 elements")
plt.plot(np.arange(500), result4[3], linewidth = 0.8, label="2 hidden layer with 128 and 32 elements")
plt.ylim(0.9, 1)
plt.xlabel("epochs")
plt.ylabel("accuracy of test data")
plt.legend()
plt.title("Compare of models with different structure")
plt.show()

#Compare of different activation functions
model5 = NeuralNetwork(dim, 512, 128, 10, sigmoid, d_sigmoid)
result5 = train_steps(model5, 0.001, 500, 256, train_in, train_out, test_in, test_out)
plt.figure()
plt.plot(np.arange(500), result3[3], label="relu")
plt.plot(np.arange(500), result5[3], label="sigmoid")
plt.ylim(0, 1)
plt.xlabel("epochs")
plt.ylabel("accuracy of test data")
plt.title("Compare of different activation functions")
plt.legend()
plt.show()

#Compare of different learning rate
model6 = NeuralNetwork(dim, 512, 128, 10, relu, d_relu)
result6 = train_steps(model6, 0.01, 10, 256, train_in, train_out, test_in, test_out)
model7 = NeuralNetwork(dim, 512, 128, 10, relu, d_relu)
result7 = train_steps(model7, 0.0001, 500, 256, train_in, train_out, test_in, test_out)
plt.figure()
plt.plot(np.arange(500), result3[0], label="learning rate=0.001")
plt.plot(np.arange(500), result7[0], label="learning rate=0.0001")
plt.xlabel("epochs")
plt.ylabel("loss of training data")
plt.title("Compare of different learning rate")
plt.legend()
plt.show()

#Plot of results
plt.figure()
plt.plot(np.arange(500), result3[0], label="training data")
plt.plot(np.arange(500), result3[1], label="test data")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("loss of optimal model")
plt.legend()
plt.show()
plt.figure()
plt.plot(np.arange(500), result3[2], label="training data")
plt.plot(np.arange(500), result3[3], label="test data")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.title("accuracy of optimal model")
plt.legend()
plt.show()
