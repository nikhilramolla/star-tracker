#/usr/bin/env python3
import torch.nn as nn
import torch.nn.functional as F

class VGG16(nn.Module):
    def __init__(self, kernel_size=3, padding_size=1, dropout_prob=0.1):
        super(VGG16, self).__init__()
        self.c1=nn.Conv2d(1,128,kernel_size,padding=padding_size)
        self.c2=nn.Conv2d(128,128,kernel_size,padding=padding_size)
        self.bn1=nn.BatchNorm2d(128)
        self.p1=nn.MaxPool2d(2,stride=2)
        
        self.c3=nn.Conv2d(128,128,kernel_size,padding=padding_size)
        self.c4=nn.Conv2d(128,128,kernel_size,padding=padding_size)
        self.bn2=nn.BatchNorm2d(128)
        self.p2=nn.MaxPool2d(2,stride=2)
        
        self.c5=nn.Conv2d(128,128,kernel_size,padding=padding_size)
        self.c6=nn.Conv2d(128,128,kernel_size,padding=padding_size)
        self.bn3=nn.BatchNorm2d(128)
        self.c7=nn.Conv2d(128,128,kernel_size,padding=padding_size)
        self.p3=nn.MaxPool2d(2,stride=2)
        
        self.c8=nn.Conv2d(128,256,kernel_size,padding=padding_size)
        self.c9=nn.Conv2d(256,256,kernel_size,padding=padding_size)
        self.bn4=nn.BatchNorm2d(256)
        self.c10=nn.Conv2d(256,256,kernel_size,padding=padding_size)
        self.p4=nn.MaxPool2d(2,stride=2)
        # add the flatten layer in the forward function of p4 itself, similar with others
        self.drop=nn.Dropout(p=dropout_prob)
        self.fc1=nn.Linear(36864,1024)
        self.bn5=nn.BatchNorm1d(1024)
        self.fc2=nn.Linear(1024,1024)
        self.fc3=nn.Linear(1024,88)
        
    
    def forward(self,x):
        x=self.bn1(self.c1(x))
        x=self.bn1(self.c2(x))
        x=self.p1(x)
        
        x=self.bn2(self.c3(x))
        x=self.bn2(self.c4(x))
        x=self.p2(x)
        
        x=self.bn3(self.c5(x))
        x=self.bn3(self.c6(x))
        x=self.bn3(self.c7(x))
        x=self.p3(x)
        
        x=self.bn4(self.c8(x))
        x=self.bn4(self.c9(x))
        x=self.bn4(self.c10(x))
        x=self.p4(x)
        
        x=x.view(x.size()[0], -1)
        x=self.bn5(F.relu(self.drop(self.fc1(x))))
        x=self.bn5(F.relu(self.drop(self.fc2(x))))
        x=F.sigmoid(self.fc3(x))
        
        return x
