# Author: Kushagra Juneja
# Description: Reads the constellation data from data.csv, processes it and then 
#              writes it to data.inc in the desired format.

import pandas as pd
from pandas import Series,DataFrame

csv_data = pd.read_csv("data.csv")

# print(csv_data.iloc[1121,0])

data=[]
name=[]
last="lol"
for row in range(7,1122):
	constellation=csv_data.iloc[row,0]
	ra=csv_data.iloc[row,2]
	dec=csv_data.iloc[row,3]
	if(pd.isnull(ra)):
		continue
	ra=float(ra)
	ra=ra*360.0/24.0
	dec=float(dec)
	if(constellation==last):
		sz=len(data)
		sz=sz-1
		data[sz][ra]=dec
	else:
		last=constellation
		data.append({})
		name.append(constellation)
		sz=len(data)
		sz=sz-1
		data[sz][ra]=dec

string="var data= {"
print(string)
for i in range(0, 88):
	arr=[]
	for key,value in data[i].items():
		arr.append([key,value])
	if(i!=87):	
		print(name[i],":",arr,",")
	else:
		print(name[i],":",arr)
print("} ;")

# for i in name:
# 	print (i)