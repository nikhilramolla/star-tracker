#!/usr/bin/env python3
import score
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
kernel_size = 7
padding_size = 3
dropout_prob = 0


#device config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
write("Testing on %s\n"%device)


#labels(Y) config
Y = [list(map(float, label.split(" ")[1:])) for label in open("../data/test/output.txt", "r").read().split("\n")[1:]]

# data generator for training
def get_image(n):
    batch = np.array(range(16*n, 16*(n+1)))
    x = np.asarray([[np.array(cv2.imread("../data/test/%d.png" %(i) ,cv2.IMREAD_GRAYSCALE)-5)/50] for i in batch])
    #print(x.shape)
    y = np.asarray([Y[i] for i in batch])
    #print(y.shape)
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
	write('Model succesfully loaded.\n')
except:
	write("Training file not found/ incompatible with model/ some other error.\n")
model.to(device)

model = model.eval()

global_y = np.asarray([])
global_y_test = np.asarray([])

for n in range(62):
        x,y = get_image(n)
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        y_pred = y_pred.cpu().detach().numpy()
        if(n == 0):
            global_y = y
            global_y_test = y_pred
        else:
            global_y = np.concatenate((global_y, y))
            global_y_test = np.concatenate((global_y_test, y_pred))

global_y = torch.tensor(np.array(global_y))
global_y_test = torch.tensor(np.array(global_y_test))

accuracy,f1 = score.evaluate(global_y, global_y_test)

write("accuracy: %s"%accuracy)
write("\n")
write("f1: %s"%f1)
end_time = time.time()
write("Time: %f seconds"%(end_time-start_time))
log.close()
