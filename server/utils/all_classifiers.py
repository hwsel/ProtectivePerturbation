from cifar10_models.densenet import densenet121, densenet161, densenet169
from cifar10_models.googlenet import googlenet
from cifar10_models.inception import inception_v3
from cifar10_models.mobilenetv2 import mobilenet_v2
from cifar10_models.resnet import resnet18, resnet34, resnet50
from cifar10_models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn


all_classifiers = {
        "vgg11_bn": vgg11_bn,
        "vgg13_bn": vgg13_bn,
        "vgg16_bn": vgg16_bn,
        "vgg19_bn": vgg19_bn,
        "resnet18": resnet18,
        "resnet34": resnet34,
        "resnet50": resnet50,
        "densenet121": densenet121,
        "densenet161": densenet161,
        "densenet169": densenet169,
        "mobilenet_v2": mobilenet_v2,
        "googlenet": googlenet,
        "inception_v3": inception_v3,
}

classifier_list = [
                   'vgg13_bn',
                   'vgg16_bn',
                   'resnet18',
                   'resnet34',
                   'densenet121',
                   'mobilenet_v2',
                   'googlenet',
                   'inception_v3'
                   ]
