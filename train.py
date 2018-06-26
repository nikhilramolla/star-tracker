#Model trainer
#Authors: Kushagra Juneja, Nikhil Reddy
print("Loading modules... ")
import torch
import torch.nn.functional as F # most non-linearities are here
import torch.optim as optim
import random
import cv2
import numpy as np
from model import VGG16 # model imported from model.py
from time import time


#primary config
epochs = 20
batches = 156
batch_size = 64
learning_rate = 1e-4
kernel_size = 3
padding_size = 1
# dropout_prob = 0.1
validate_stride = 10


#device config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Training on", device)


#labels(Y) config
print("Loading labels... ")
Y = open("../data/output.txt", "r").read().split("\n")[1:]
Y = [list(map(float, label.split(" ")[1:])) for label in Y]


# testing config
print("Loading test cases... ")
J_test = np.random.choice(10000, batch_size, replace = False)
x_test = [(np.array(cv2.imread("../data/small/%d.png"%j, cv2.IMREAD_GRAYSCALE))-5)/50 for j in J_test]
x_test = np.resize(np.asarray(x_test),(batch_size,1,144,256))
x_test = torch.tensor(x_test)
x_test = x_test.float()
x_test = x_test.to(device)
y_test = [Y[j] for j in J_test]
y_test = torch.tensor(y_test)
y_test = y_test.float()
y_test = y_test.to(device)


# data generator for training
def generate_data():
    x,y = [],[]
    for i in range(batch_size):
        j = random.randint(0,9999)
        while j in J_test:
            j = random.randint(0,10000)
        img = np.array(cv2.imread("../data/small/%d.png" %j ,cv2.IMREAD_GRAYSCALE))
        img = (img-5)/50
        x.append(img)
        y.append(Y[j])
    x = np.resize(np.asarray(x), (batch_size,1,144,256))
    x = torch.tensor(x)
    x = x.float()
    y = np.asarray(y)
    y = torch.tensor(y)
    y = y.float()
    return x,y

print("Loading model... ")
model = VGG16(kernel_size=kernel_size, padding_size=padding_size)
try:
	model.load_state_dict(torch.load('../data/mytraining.pt'))
except:
	print("Training file not found/ incompatible with model/ some other error.")
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("Initiating sequence...")
for epoch in range(epochs):
    for batch in range(batches):
        print("### Batch %3d of Epoch %3d" % (batch, epoch), end="")
        x,y = generate_data()
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        optimizer.step()
        if(batch%validate_stride==0):
            y_pred=model(x_test)
            loss = 0
            for i in range(88):
                loss += F.binary_cross_entropy(y_pred[:,i], y[:,i])
            loss /= 88.0
            print(", ", loss, end="")
        y_pred = model(x)
        loss = 0
        for i in range(88):
            loss += F.binary_cross_entropy(y_pred[:,i], y[:,i])
        loss /= 88.0
        print(", ", loss)
    print("Saving state... ", end="")         
    torch.save(model.state_dict(), '../data/mytraining.pt')
    print("done.")
    print("### Epoch completed")
print("Sequence completed")
