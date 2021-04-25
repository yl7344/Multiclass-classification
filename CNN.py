import torch.nn as nn

# construct the network
class CNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=5) 
        self.BN1 = nn.BatchNorm2d(4)
        self.pool = nn.MaxPool2d(2, 2) 
        self.conv2 = nn.Conv2d(4, 16, kernel_size=3)
        self.BN2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 64, kernel_size=3)
        self.BN3 = nn.BatchNorm2d(64)          
        self.conv4 = nn.Conv2d(64, 256, kernel_size=3)
        self.BN4 = nn.BatchNorm2d(256)         
        self.fc1 = nn.Linear(3*3*256, num_classes)
        #self.fc2 = nn.Linear(1024, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)
    
    def forward(self, x):
        out = self.pool(self.relu(self.BN1(self.conv1(x))))
        out = self.relu(self.BN2(self.conv2(out)))
        out = self.pool(self.relu(self.BN3(self.conv3(out))))
        out = self.relu(self.BN4(self.conv4(out)))
        #out = self.relu(self.BN5(self.conv5(out)))
        #out = self.pool(self.relu(self.conv1(x)))
        #print(out.shape)
        #out = self.pool(self.relu(self.conv2(out)))
        #print(out.shape)
        #out = self.pool(self.relu(self.conv3(out)))
        #print(out.shape)
        out = out.view(-1,3*3*256)
        out = self.softmax(self.fc1(out))
        #out = self.fc2(out)
        return out

def buildNetwork(num_classes):
    return CNN((3,32,32),num_classes)