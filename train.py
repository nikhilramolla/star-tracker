#/usr/bin/env python3
import sys
import time
log = open("train.log", "a+")
def write(x):
	log.write(x)
	sys.stdout.write(x)
start_time = time.time()

#modules
import torch
import torch.nn.functional as F # most non-linearities are here
import torch.optim as optim
import random
import cv2
import numpy as np
from model import VGG16 # model imported from model.py


#primary config
epochs = 20
batches = 156
batch_size = 64
learning_rate = 1e-4
kernel_size = 3
padding_size = 1
dropout_prob = 0
dataset_size = {
    "fixed": 10000,
    "test": 2000
}


#device config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
write("Training on %s\n"%device)


#labels(Y) config
Y = {dataset: [list(map(float, label.split(" ")[1:])) for label in open("../data/%s/output.txt"%dataset, "r").read().split("\n")[1:]] for dataset in dataset_size}

# data generator for training
def generate_data(dataset):
    x,y = [],[]
    batch = np.random.choice(dataset_size[dataset], batch_size)
    x = np.asarray([(np.array(cv2.imread("../data/%s/%d.png" %(dataset, i) ,cv2.IMREAD_GRAYSCALE)-5)/50 for i in batch])
    y = [Y[dataset][i] for i in batch]
    x = torch.tensor(x)
    x = x.float()
    y = np.asarray(y)
    y = torch.tensor(y)
    y = y.float()
    return x,y

model = VGG16(
    kernel_size=kernel_size,
    padding_size=padding_size,
    dropout_prob=dropout_prob
)
try:
	model.load_state_dict(torch.load('../data/mytraining.pt'))
	write('Model succesfully loaded.')
except:
	write("Training file not found/ incompatible with model/ some other error.\n")
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    for batch in range(batches):
        write("%3d %3d"%(batch, epoch))
        x,y = generate_data()
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        y_pred = model(x)
        optimizer.zero_grad()
        optimizer.step()
        loss = 0
        y_pred=model(x_test)
        for i in range(88):
            loss += F.binary_cross_entropy(y_pred[:,i], y[:,i])
        loss /= 88.0
        write(" %s\n"%str(loss)[7:13]
    torch.save(model.state_dict(), '../data/mytraining.pt')
end_time = time.time()
write("Time: %f seconds"%(end_time-start_time))
log.close()
