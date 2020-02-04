import torch
from torch import nn

class LinearPilot(nn.Module):
    def __init__(self, output_ch=1, stochastic=True, steer_bins=None, throttle_bins=None):
        super().__init__()
        self.output_ch = output_ch
        self.stochastic = stochastic
        self.steer_bins = steer_bins
        self.throttle_bins = throttle_bins
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
            self.fc_throttle = nn.Linear(50, len(self.throttle_bins))
            self.fc_steer = nn.Linear(50, len(self.steer_bins))
        elif self.output_ch == 1:
            self.fc_steer = nn.Linear(50,len(self.steer_bins))
    
    

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
            throttle_logits = self.fc_throttle(x)
            steer_logits = self.fc_steer(x)
            return throttle_logits, steer_logits
        elif self.output_ch == 1:
            steer_logits = self.fc_steer(x)
            return steer_logits