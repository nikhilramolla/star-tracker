#!/usr/bin/env python3
import numpy as np
import pandas as pd
from torch import cuda
import numba as nb
from torch import nn

'''
Architectue Layout: The Star tracker model
'''

class Tracker(nn.module):
	'''The Star Tracker ConvNet class'''
	def __init__(self):
		'''Initializes the instance and defines the architecture'''
		super(Tracker, self).__init__()
	def train(self, data, promise):
		'''Train the network with the data'''
	def test(self, data, promise):
		'''Test the network with the data'''
	def save(self, name, promise):
		'''Save the model on storage'''
	def load(self, name, promise):
		'''Load the model from storage'''