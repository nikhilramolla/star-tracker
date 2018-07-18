# Take care to caluculate f1_score and accuracy_score for large data(atleast around 100 samples)
# This can be done by concatenating y_true and y_pred for various iterations since RAM may not be enough for such large batch size.

import numpy as np

def round(arr):
	return (np.array(arr)>0.5).astype(np.int)

# print(round([[0.9,0.1],[0.2,0.8]]))

def accuracy_score(a,b):
	return sum((np.array(a)==np.array(b)).astype(np.int))/len(a)

# print(accuracy_score([1,2,3],[1,4,2]))

def f1_score(y_true,y_pred):
	#print(y_true,y_pred)
	y_true=np.array(y_true)
	y_pred=np.array(y_pred)
	true_pos=sum(np.logical_and(y_true==1,y_pred==1))
	false_pos=sum(np.logical_and(y_true==0,y_pred==1))
	false_neg=sum(np.logical_and(y_true==1,y_pred==0))
	true_neg=sum(np.logical_and(y_true==0,y_pred==0))
	precision=true_pos/(true_pos+false_pos)
	recall=true_pos/(true_pos+false_neg)
	f1=2*precision*recall/(precision+recall)
	return f1


# returns two array of 88 numbers i.e. accuracy,f1_score
def evaluate(y_true,y_pred):
	accuracy=[]
	f1=[]
	y_true=np.array(round(y_true))
	y_pred=np.array(round(y_pred))
	for i in range(0, 88):
		# print(i)
		accuracy.append(accuracy_score(y_true[:,i],y_pred[:,i]))
		f1.append(f1_score(y_true[:,i],y_pred[:,i]))
	return accuracy,f1

# print(evaluate([[0.1,0.1],[0.9,0.9]],[[0.2,0.2],[0.8,0.8]]))
