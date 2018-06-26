import torch.nn as nn
import torch.nn.functional as F

class VGG16(nn.Module):
    def __init__(self, kernel_size=3, padding_size=1):
        super(VGG16, self).__init__()
        self.c1 = nn.Conv2d(1,32,kernel_size=kernel_size,padding=padding_size)
        self.c2 = nn.Conv2d(32,32,kernel_size=kernel_size,padding=padding_size)
        self.p1 = nn.MaxPool2d(2,stride=2)
        
        self.c3 = nn.Conv2d(32,64,kernel_size=kernel_size,padding=padding_size)
        self.c4 = nn.Conv2d(64,64,kernel_size=kernel_size,padding=padding_size)
        self.p2 = nn.MaxPool2d(2,stride=2)
        
        self.c5 = nn.Conv2d(64,128,kernel_size=kernel_size,padding=padding_size)
        self.c6 = nn.Conv2d(128,128,kernel_size=kernel_size,padding=padding_size)
        self.c7 = nn.Conv2d(128,128,kernel_size=kernel_size,padding=padding_size)
        self.p3 = nn.MaxPool2d(2,stride=2)
        
        self.c8 = nn.Conv2d(128,256,kernel_size=kernel_size,padding=padding_size)
        self.c9 = nn.Conv2d(256,256,kernel_size=kernel_size,padding=padding_size)
        self.c10 = nn.Conv2d(256,256,kernel_size=kernel_size,padding=padding_size)
        self.p4 = nn.MaxPool2d(2,stride=2)
        
        self.fc1 = nn.Linear(36864,1024)
        self.fc2 = nn.Linear(1024,1024)
        self.fc3 = nn.Linear(1024,88)
    
    def forward(self,x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.p1(x)
        
        x = self.c3(x)
        x = self.c4(x)
        x = self.p2(x)
        
        x = self.c5(x)
        x = self.c6(x)
        x = self.c7(x)
        x = self.p3(x)
        
        x = self.c8(x)
        x = self.c9(x)
        x = self.c10(x)
        x = self.p4(x)
        
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        
        return x
