import torch
from torch import nn

class LinearPilot(nn.Module):
    def __init__(self, output_ch=2):
        super().__init__()
        self.output_ch = output_ch
        self.bn = nn.BatchNorm2d(6)
        self.conv2d_1 = nn.Sequential(
                            nn.Conv2d(6, 24,kernel_size=5, stride=2, padding=0,),
                            nn.ReLU(),
                            nn.Dropout2d(p=0.1))
        self.conv2d_2 = nn.Sequential(
                            nn.Conv2d(24, 32,kernel_size=5, stride=2, padding=0,),
                            nn.ReLU(),
                            nn.Dropout2d(p=0.1))
        self.conv2d_3 = nn.Sequential(
                            nn.Conv2d(32, 64,kernel_size=5, stride=2, padding=0,),
                            nn.ReLU(),
                            nn.Dropout2d(p=0.1))
        self.conv2d_4 = nn.Sequential(
                            nn.Conv2d(64, 64,kernel_size=3, stride=1, padding=0,),
                            nn.ReLU(),
                            nn.Dropout2d(p=0.1))
        self.conv2d_5 = nn.Sequential(
                            nn.Conv2d(64, 64,kernel_size=3, stride=1, padding=0,),
                            nn.ReLU(),
                            nn.Dropout2d(p=0.1))
        
        in_size=2496
        # FC1
        self.fc1 = nn.Sequential(
            nn.Linear(in_size, 100),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )

        # FC2
        self.fc2 = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        ) 
        if self.output_ch == 2:
            self.fc_throttle = nn.Linear(50, 1)
            self.fc_steer = nn.Linear(50,1)
        elif self.output_ch == 1:
            self.fc_steer = nn.Linear(50,1)
    
    

    def forward(self, img):
        batch = img.shape[0]
        x = self.bn(img)
        x = self.conv2d_1(x)
        x = self.conv2d_2(x)
        x = self.conv2d_3(x)
        x = self.conv2d_4(x)
        x = self.conv2d_5(x)
        x = x.view(batch, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        if self.output_ch == 2:
            throttle = self.fc_throttle(x).view(-1)
            steer = self.fc_steer(x).view(-1)
            return throttle, steer
        elif self.output_ch == 1:
            steer = self.fc_steer(x).view(-1)
            return steer
