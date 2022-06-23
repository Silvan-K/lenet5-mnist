from torch import nn

class LeNet5(nn.Module):
    
    def __init__(self):
        
        super().__init__()

        self.layer0 = nn.Sequential(nn.Conv2d(1, 6, kernel_size=5),
                                    nn.BatchNorm2d(6),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size = 2,
                                                 stride = 2))
        
        self.layer1 = nn.Sequential(nn.Conv2d(6, 16, kernel_size=5,
                                              stride=1, padding=0),
                                    nn.BatchNorm2d(16),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size = 2,
                                                 stride = 2))
        
        self.fc0 = nn.Linear(400, 120)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84,  10)
        
    def forward(self, x):
        out = self.layer0(x)
        out = self.layer1(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc0(out).relu()
        out = self.fc1(out).relu()
        out = self.fc2(out)
        return out
