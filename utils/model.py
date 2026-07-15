import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class SiameseDamageNet(nn.Module):
    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        self.feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.head = nn.Sequential(
            nn.Linear(self.feature_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, pre_img: torch.Tensor, post_img: torch.Tensor) -> torch.Tensor:
        pre_feat = self.backbone(pre_img)
        post_feat = self.backbone(post_img)
        merged = torch.cat([pre_feat, post_feat], dim=1)
        return self.head(merged)
