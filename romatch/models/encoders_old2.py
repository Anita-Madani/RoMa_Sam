import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from romatch.utils.utils import get_autocast_params
from transformers import SamModel
import torch.distributed as dist

class ResNet50(nn.Module):
    def __init__(self, pretrained=False, high_res=False, weights=None, dilation=None,
                 freeze_bn=True, anti_aliased=False, early_exit=False, amp=False, amp_dtype=torch.float16):
        super().__init__()
        if dilation is None:
            dilation = [False, False, False]
        if weights is not None:
            self.net = tvm.resnet50(weights=weights, replace_stride_with_dilation=dilation)
        else:
            self.net = tvm.resnet50(pretrained=pretrained, replace_stride_with_dilation=dilation)

        self.high_res = high_res
        self.freeze_bn = freeze_bn
        self.early_exit = early_exit
        self.amp = amp
        self.amp_dtype = amp_dtype

    def forward(self, x, **kwargs):
        autocast_device, autocast_enabled, autocast_dtype = get_autocast_params(x.device, self.amp, self.amp_dtype)
        with torch.autocast(autocast_device, enabled=autocast_enabled, dtype=autocast_dtype):
            net = self.net
            feats = {1: x}
            x = net.conv1(x)
            x = net.bn1(x)
            x = net.relu(x)
            feats[2] = x
            x = net.maxpool(x)
            x = net.layer1(x)
            feats[4] = x
            x = net.layer2(x)
            feats[8] = x
            if self.early_exit:
                return feats
            x = net.layer3(x)
            feats[16] = x
            x = net.layer4(x)
            feats[32] = x
            return feats

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

class CNNandSAM(nn.Module):
    def __init__(self, cnn_kwargs=None, amp=False, amp_dtype=torch.float16, use_vgg=False):
        super().__init__()

        print("Loading SAM model from Hugging Face...")
        sam_model_name = "facebook/sam-vit-base"

        if not dist.is_initialized() or dist.get_rank() == 0:
            sam_encoder = SamModel.from_pretrained(sam_model_name)
            sam_encoder.eval()
            self.sam_encoder = [sam_encoder]  # DDP-safe pattern
        else:
            self.sam_encoder = [None]

        self.amp = amp
        self.amp_dtype = amp_dtype
        cnn_kwargs = cnn_kwargs if cnn_kwargs is not None else {}
        self.cnn = ResNet50(**cnn_kwargs)
        self.use_sam = True

    def train(self, mode: bool = True):
        return self.cnn.train(mode)

    def forward(self, x, upsample=False):
        B, C, H, W = x.shape
        feature_pyramid = self.cnn(x)

        if not upsample and self.sam_encoder[0] is not None:
            x_sam = F.interpolate(x, size=(1024, 1024), mode="bilinear", align_corners=False)
            with torch.no_grad():
                self.sam_encoder[0] = self.sam_encoder[0].to(x.device)

                # 👇 This is the correct way to get dense embeddings
                dense_embeddings = self.sam_encoder[0].vision_encoder(x_sam)

                feats_16 = dense_embeddings  # [B, C, H, W]
                print("shape of features ", feats_16.shape)

                feature_pyramid[16] = feats_16
                for feat in feature_pyramid:
                    print("shape of features ", feat.shape )

        return feature_pyramid
