import torch
from torch import nn

class DcieLoss(nn.Module):
    def __init__(self):
        super(DcieLoss,self).__init__()
        self.smooth = 1e-5
        # self.sigmoid = nn.Sigmoid()

    def forward(self,X,mask):
        # X = self.sigmoid(X)
        intersection = torch.sum(X*mask)
        union = torch.sum(X) + torch.sum(mask)
        dice = (2*intersection+self.smooth)/(union+self.smooth)

        return 1-dice