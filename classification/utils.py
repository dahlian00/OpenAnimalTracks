
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import swin_b, Swin_B_Weights
from torchvision.models import vit_b_32, ViT_B_32_Weights
from torchvision.models import vgg16, VGG16_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
from torchvision.models import inception_v3, Inception_V3_Weights
from torch import nn

def get_models(name,n_classes,return_head=False):
    if name=='resnet18':
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
        head=model.fc
    elif name=='resnet50':
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
        head=model.fc
    elif name=='inception_v3':
        weights = Inception_V3_Weights.DEFAULT
        model = inception_v3(weights=weights)
        model.fc = inception_v3(num_classes=n_classes).fc
        head=model.fc
    elif name=='swin_b':
        weights = Swin_B_Weights.DEFAULT
        model = swin_b(weights=weights)
        model.head = nn.Linear(model.head.in_features, n_classes)
        head=model.head
    elif name=='vit_b':
        weights = ViT_B_32_Weights.DEFAULT
        model = vit_b_32(weights=weights)
        model.heads = vit_b_32(num_classes=n_classes).heads
        head=model.heads
    elif name=='vgg16':
        weights = VGG16_Weights.DEFAULT
        model = vgg16(weights=weights)
        model.classifier = vgg16(num_classes=n_classes).classifier
        head=model.classifier
    elif name=='efficientnet_b0':
        weights = EfficientNet_B0_Weights.DEFAULT
        model = efficientnet_b0(weights=weights)
        model.classifier = efficientnet_b0(num_classes=n_classes).classifier
        head=model.classifier
    elif name=='efficientnet_b1':
        weights = EfficientNet_B1_Weights.DEFAULT
        model = efficientnet_b1(weights=weights)
        model.classifier = efficientnet_b1(num_classes=n_classes).classifier
        head=model.classifier
    else:
        raise NotImplementedError
    
    if return_head:
        return model,head
    else:
        return model