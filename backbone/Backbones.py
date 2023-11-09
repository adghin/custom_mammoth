"""
@author: adrian.ghinea@outlook.it
Use this file for loading and adapting pre-trained pytorch models as backbones for experiments on complex datasets
"""
import torch.nn as nn
import torchvision.models as models

def custom_network(model,dataset):
    """
    Apply changes to the backbone's pre-trained network architecture in order to perform experiments on datasets different
    from the original one (i.e. ImageNet-1K)
    """

    if(dataset == 'seq-cifar10'):
        out_classes = 10
    if(dataset == 'seq-cifar100'):
        out_classes = 100
    
    #Look after dataset == seq-tinyimg

    #CIFAR-10: we need to change the first "conv1" layer and the last "fc" layer

    #Changing "conv1" layer from kernel_size=7, stride=2, padding=3 --> kernel_size=3, stride=1, padding=1
    in_channels     = 3     #as per default
    inplanes        = 64    #as per default
    new_kernel_size = 3     #changed
    new_stride      = 1     #changed
    new_padding     = 1     #changed

    model.conv1 = nn.Conv2d(in_channels, inplanes, kernel_size=new_kernel_size, stride=new_stride, padding=new_padding, bias=False)

    #Changing "fc" layer according to the number of classes: from num=1000 --> num=10
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, out_classes)

    return model

def get_backbone(model,dataset):
    """
    Load pre-trained model with default weights from torchvision.models
    :param model: model name to load from pytorch
    :param dataset: dataset needed in order to apply the correct changes to the network architecture
    :return: New model network architecture
    """
    model_name = model
    model_weights = "DEFAULT"
    model = models.get_model(model_name,weights=model_weights)

    if(dataset == 'seq-cifar10' or dataset == 'seq-cifar100'):
        adapted_model = custom_network(model,dataset)
        return adapted_model

    #implement seq-tinyimg (highres) and (nonhighres)
    return model