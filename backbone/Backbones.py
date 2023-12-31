"""
@author: adrian.ghinea@outlook.it
Use this file for loading and adapting pre-trained pytorch models as backbones for experiments on complex datasets
"""
import torch
import torch.nn as nn
import torchvision.models as models

def custom_resnet(model,dataset,out_classes,upscale):
    """
    Implement changes for custom resnet model
    :param model: resnet model
    :param dataset: dataset to use
    :param out_classes: number of output classes
    :return: custom resnet model
    """
    #We need to change the first "conv1" layer (if dataset is not TINYIMG-HD) and the last "fc" layer

    #Changing "conv1" layer params from kernel_size=7, stride=2, padding=3 --> kernel_size=3, stride=1, padding=1
    in_channels     = 3     #as per default
    inplanes        = 64    #as per default
    new_kernel_size = 3     #changed
    new_stride      = 1     #changed
    new_padding     = 1     #changed

    #Change conv1 layer if dataset ***is not*** TINYIMGNET-HD or images ***are not*** upscaled to model's default res
    if(upscale == 0 and dataset != 'seq-tinyimg-hd'):
        model.conv1 = nn.Conv2d(in_channels, inplanes, kernel_size=new_kernel_size, stride=new_stride, padding=new_padding, bias=False)

    #Changing "fc" layer according to the number of datasets' classes
    num_features    = model.fc.in_features
    model.fc        = nn.Linear(num_features, out_classes)

    return model

def custom_vit(model,dataset,out_classes,upscale):
    """
    Implement changes for custom vit model
    :param model: vit model
    :param dataset: dataset to use
    :param out_classes: number of output classes
    :return: custom vit model
    """

    from functools import partial
    from collections import OrderedDict

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
    
    #Apply changes to "conv_proj" and sequence_length if dataset ***is not*** TINYIMG-HD and images ***are not*** upscaled to model's default res
    if(upscale == 0 and dataset != 'seq-tinyimg-hd'):
        model.image_size    = image_size
        model.patch_size    = patch_size
        model.conv_proj     = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=model.patch_size,stride=model.patch_size)

        seq_length          = (model.image_size // model.patch_size) ** 2
        seq_length         += 1

        num_layers          = 12
        num_heads           = 12
        hidden_dim          = 768
        mlp_dim             = 3072
        dropout             = 0.0
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
    
def custom_network(model_name,model,dataset,upscale):
    """
    Apply changes to the backbone's pre-trained network architecture in order to perform experiments on datasets different
    from the original one (i.e. ImageNet-1K)
    """

    if('seq-cifar10' in dataset):
        out_classes = 10
    if('seq-cifar100' in dataset):
        out_classes = 100
    if('seq-tinyimg' in dataset or 'seq-imagenetR' in dataset):
        out_classes = 200

    if("resnet" in model_name):
        model = custom_resnet(model,dataset,out_classes,upscale)
    if("vit" in model_name):
        model = custom_vit(model,dataset,out_classes,upscale)

    return model

def get_backbone(backbone,dataset,upscale):
    """
    Load pre-trained model with default weights from torchvision.models
    :param model: model name to load from pytorch
    :param dataset: dataset needed in order to apply the correct changes to the network architecture
    :return: New model network architecture
    """
    model_name     = backbone
    model_weights  = "DEFAULT"
    model          = models.get_model(model_name,weights=model_weights)

    adapted_model  = custom_network(model_name,model,dataset,upscale)
    return adapted_model
