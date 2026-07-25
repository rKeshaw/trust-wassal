# resnet_caltech.py
import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18Caltech(nn.Module):
    """
    ResNet18 for Caltech-101 (224x224 images, 102 classes)
    Compatible with gradient extraction for active learning strategies
    """
    def __init__(self, num_classes=102, pretrained=False):
        super(ResNet18Caltech, self).__init__()
        
        # Load pretrained or random initialization
        if pretrained:
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            print("✓ Loaded pretrained ImageNet weights for ResNet18")
        else:
            self.model = models.resnet18(weights=None)
            print("✓ Initialized ResNet18 with random weights")
        
        # Replace final layer for Caltech-101 classes
        in_features = self.model.fc.in_features  # 512 for ResNet18
        self.model.fc = nn.Linear(in_features, num_classes)
        
        # Store embedding dimension for active learning strategies
        self.embedding_dim = in_features
        self.num_classes = num_classes
        
        # CRITICAL FIX: Expose internal layers for WASSAL
        self.conv1 = self.model.conv1
        self.bn1 = self.model.bn1
        self.relu = self.model.relu
        self.maxpool = self.model.maxpool
        self.layer1 = self.model.layer1
        self.layer2 = self.model.layer2
        # self.layer3 = self.model.layer3
        # self.layer4 = self.model.layer4
        self.avgpool = self.model.avgpool
        self.fc = self.model.fc

    def forward(self, x):
        return self.model(x)
    
    def get_embedding_dim(self):
        """Return embedding dimension (needed for some AL strategies)"""
        return self.embedding_dim
    
    def get_features(self, x):
        """
        Extract features before final FC layer
        Useful for feature-based embeddings
        """
        # Forward through all layers except final FC
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        # x = self.model.layer3(x)
        # x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x


class ResNet50Caltech(nn.Module):
    """
    ResNet50 for Caltech-101 (224x224 images, 102 classes)
    Larger model for improved performance
    """
    def __init__(self, num_classes=102, pretrained=False):
        super(ResNet50Caltech, self).__init__()
        
        # Load pretrained or random initialization
        if pretrained:
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            print("✓ Loaded pretrained ImageNet weights for ResNet50")
        else:
            self.model = models.resnet50(weights=None)
            print("✓ Initialized ResNet50 with random weights")
        
        # Replace final layer for Caltech-101 classes
        in_features = self.model.fc.in_features  # 2048 for ResNet50
        self.model.fc = nn.Linear(in_features, num_classes)
        
        # Store embedding dimension
        self.embedding_dim = in_features
        self.num_classes = num_classes
        
        # CRITICAL FIX: Expose internal layers for WASSAL
        self.conv1 = self.model.conv1
        self.bn1 = self.model.bn1
        self.relu = self.model.relu
        self.maxpool = self.model.maxpool
        self.layer1 = self.model.layer1
        self.layer2 = self.model.layer2
        # self.layer3 = self.model.layer3
        # self.layer4 = self.model.layer4
        self.avgpool = self.model.avgpool
        self.fc = self.model.fc

    def forward(self, x):
        return self.model(x)
    
    def get_embedding_dim(self):
        """Return embedding dimension"""
        return self.embedding_dim
    
    def get_features(self, x):
        """Extract features before final FC layer"""
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        # x = self.model.layer3(x)
        # x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x


def get_model(num_classes=102, model_name='ResNet18', pretrained=False):
    """Factory function to create ResNet models for Caltech-101"""
    if model_name == 'ResNet18':
        return ResNet18Caltech(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'ResNet50':
        return ResNet50Caltech(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


# For backward compatibility
ResNet18 = ResNet18Caltech
ResNet50 = ResNet50Caltech