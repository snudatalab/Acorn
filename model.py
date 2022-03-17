import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, num_classes)        

        self.bn1 = nn.BatchNorm1d(100)        

    def forward(self, x):
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.fc2(out)
        return out

