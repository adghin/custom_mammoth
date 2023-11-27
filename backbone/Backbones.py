"""
@author: adrian.ghinea@outlook.it
Use this file for loading and adapting pre-trained pytorch models as backbones for experiments on complex datasets
"""
import torch
import torch.nn as nn
import torchvision.models as models

from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description='mammoth', allow_abbrev=False)

    #To use this argument add the same in utils/args.py --> add_management_args
    parser.add_argument('--optim_upscale',type=int,help='Upscale images to model's default size. Default = 0 (no upscale), 1 (upscale)',default=0,choices=[0,1])
  
def custom_resnet(model,dataset,out_classes):
    """
    Implement changes for custom resnet model
    :param model: resnet model
    :param dataset: dataset to use
    :param out_classes: number of output classes
    :return: custom resnet model
    """

    args = parse_args()
    #We need to change the first "conv1" layer (if dataset is not TINYIMG-HD) and the last "fc" layer

    #Changing "conv1" layer params from kernel_size=7, stride=2, padding=3 --> kernel_size=3, stride=1, padding=1
    in_channels     = 3     #as per default
    inplanes        = 64    #as per default
    new_kernel_size = 3     #changed
    new_stride      = 1     #changed
    new_padding     = 1     #changed

    #Change conv1 layer if dataset ***is not*** TINYIMGNET-HD or images ***are not*** upscaled to model's default res
    if(dataset != 'seq-tinyimg-hd' or args.optim_upscale != 1):
        model.conv1 = nn.Conv2d(in_channels, inplanes, kernel_size=new_kernel_size, stride=new_stride, padding=new_padding, bias=False)

    #Changing "fc" layer according to the number of datasets' classes
    num_features = model.fc.in_features
    model.fc     = nn.Linear(num_features, out_classes)

    return model

def custom_vit(model,dataset,out_classes):
    """
    Implement changes for custom vit model
    :param model: vit model
    :param dataset: dataset to use
    :param out_classes: number of output classes
    :return: custom vit model
    """

    from functools import partial
    from collections import OrderedDict

    args = parse_args()

    #For small-complex datasets such as CIFAR-10, CIFAR-100 or TINYIMG-NOHD with image resolution
    #at 32x32 or 64x64 we need to reduce the patch size (originally at patch_size=16)

    #For images at 32x32 on CIFAR-10 & CIFAR-100:
    #from patch_size = 16 to patch_size = 4
    #from image_size = 224 to image_size = 32
    cifar_patch_size = 4
    cifar_image_size = 32

    #For images at 64x64 on TINYIMG-NOHD:
    #from patch_size = 16 to patch_size = 8
    #from image_size = 224 to image_size = 64
    tinyimg_patch_size = 8
    tinyimg_image_size = 64

    if(dataset == 'seq-cifar10' or dataset == 'seq-cifar100'):
        patch_size = cifar_patch_size
        image_size = cifar_image_size
    if(dataset == 'seq-tinyimg'):
        patch_size = tinyimg_patch_size
        image_size = tinyimg_image_size

    in_channels    = 3      #as per default
    out_channels   = 768    #as per default
    
    #Apply changes to "conv_proj" and sequence_length if dataset ***is not*** TINYIMG-HD or images ***are not*** upscaled to model's default res
    if(dataset != 'seq-tinyimg-hd' or args.optim_upscale != 1):
        model.image_size    = image_size
        model.patch_size    = patch_size
        model.conv_proj     = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=model.patch_size,stride=model.patch_size)

        seq_length          = (model.image_size // model.patch_size) ** 2
        seq_length         += 1

        num_layers          = 12
        num_heads           = 12
        hidden_dim          = 768
        mlp_dim             = 3072
        attention_dropout   = 0.0
        norm_layer          = partial(nn.LayerNorm,eps=1e-6)
        
        model.encoder       = models.vision_transformer.Encoder(
                                    seq_length,
                                    num_layers,
                                    num_heads,
                                    hidden_dim,
                                    mlp_dim,
                                    dropout,
                                    attention_dropout,
                                    norm_layer,     
                                    )

        model.seq_length    = seq_length

    hidden_dim          = model.hidden_dim
    num_classes         = out_classes
    head_layers         = OrderedDict()
    head_layers["head"] = nn.Linear(hidden_dim,num_classes)

    model.heads         = nn.Sequential(head_layers)

    return model
    
def custom_network(model_name,model,dataset):
    """
    Apply changes to the backbone's pre-trained network architecture in order to perform experiments on datasets different
    from the original one (i.e. ImageNet-1K)
    """

    if(dataset == 'seq-cifar10'):
        out_classes = 10
    if(dataset == 'seq-cifar100'):
        out_classes = 100
    if(dataset == 'seq-tinyimg-hd' or dataset == 'seq-tinyimg'):
        out_classes = 200

    if("resnet" in model_name):
        model = custom_resnet(model,dataset,out_classes)
    if("vit" in model_name):
        model = custom_vit(model,dataset,out_classes)

    return model

def get_backbone(backbone,dataset):
    """
    Load pre-trained model with default weights from torchvision.models
    :param model: model name to load from pytorch
    :param dataset: dataset needed in order to apply the correct changes to the network architecture
    :return: New model network architecture
    """
    model_name     = backbone
    model_weights  = "DEFAULT"
    model          = models.get_model(model_name,weights=model_weights)

    adapted_model  = custom_network(model_name,model,dataset)
    return adapted_model

def _process_input(self, x: torch.Tensor) -> torch.Tensor:
    n, c, h, w = x.shape
    p = self.patch_size

    n_h = h // p
    n_w = w // p

    # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
    x = self.conv_proj(x)
    # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
    x = x.reshape(n, self.hidden_dim, n_h * n_w)

    # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
    # The self attention layer expects inputs in the format (N, S, E)
    # where S is the source sequence length, N is the batch size, E is the
    # embedding dimension
    x = x.permute(0, 2, 1)

    return x
