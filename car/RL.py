import torch


class PolicyNet(torch.nn.Module):
    def __init__(state_size):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_size, 10)
        self.fc2 = torch.nn.Linear(10, 5)
        self.fc3 = torch.nn.Linear(5, 2)
        self.relu = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(p=0.1)
    
    def 