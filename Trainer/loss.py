import torch.nn as nn
from torchvision.models import resnet50
import config

# phi_5,4 5th conv layer before maxpooling but after activation

class ResLoss(nn.Module):
    def __init__(self):
        super().__init__()
        model = resnet50(pretrained=True)
        names = []
        #changing last activation to identity
        for name, module in model.named_modules():
            if (hasattr(module, 'relu') and name =='layer4.2'):
                module.relu = nn.Identity()
        #removing last 2 layers
        self.resnet = nn.Sequential(*(list(model.children())[:-2])).eval().to(config.DEVICE)
        self.loss = nn.MSELoss()

        for param in self.resnet.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        res_input_features = self.resnet(input)
        res_target_features = self.resnet(target)
        return self.loss(res_input_features, res_target_features)
