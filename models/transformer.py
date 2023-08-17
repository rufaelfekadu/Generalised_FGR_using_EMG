import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# transformer class defination
model = models.vision_transformer.VisionTransformer(image_size=4, patch_size=1, num_classes=10, hidden_dim=20, num_layers=3, num_heads=2, mlp_dim=40, dropout=0.1, attention_dropout=0.1)

print(model)

#number of parameters
pytorch_total_params = sum(p.numel() for p in model.parameters())
print(pytorch_total_params)

class vision(models.vision_transformer.VisionTransformer):
    def __init__(self, image_size, patch_size, num_classes, hidden_dim, num_layers, num_heads, mlp_dim, attention_dropout):
        super().__init__(image_size=image_size,
                        patch_size = patch_size,
                        num_classes = num_classes,
                        hidden_dim=hidden_dim,
                        num_layers=num_layers, 
                        num_heads=num_heads, 
                        mlp_dim=mlp_dim,
                        attention_dropout=attention_dropout)
        self.conv_proj = nn.Conv2d(
                in_channels=1, out_channels=64, kernel_size=2, stride=2
            )