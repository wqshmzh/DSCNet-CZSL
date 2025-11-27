'''
Image feature extractor
'''
from torchvision.models import resnet18, ResNet18_Weights

class ResNet18():
    def __init__(self, device, weights=ResNet18_Weights.IMAGENET1K_V1):
        self.resnet = resnet18(weights=weights).to(device)
        
    def __call__(self, x):
        # change forward here
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x_features = self.resnet.layer4(x)
        
        return x_features
    
    def parameters(self):
        return self.resnet.parameters()
    
    def eval(self):
        self.resnet.eval()