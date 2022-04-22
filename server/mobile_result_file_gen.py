import os

import torch
from torch.nn import Softmax
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from pytorch_lightning import seed_everything
from pytorch_lightning.metrics import Accuracy

from utils.all_classifiers import all_classifiers
from utils.utils import unnormalize
from mobile_data import C10IMGDATA
from utils.ssim import SSIM


def get_args():
    parser = ArgumentParser()
    seed_everything(0)
    parser.add_argument("--data_dir", type=str, default="./mobile_protected_img")
    parser.add_argument("--model_A", type=str, default="googlenet")
    parser.add_argument("--model_B", type=str, default="vgg13_bn")
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--precision", type=int, default=16, choices=[16, 32])
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--gpu_id", type=str, default="0")
    return parser.parse_args()


if __name__ == '__main__':
    seed_everything(0)
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    ssim_calc = SSIM(data_range=1, size_average=True)
    m = Softmax(dim=1)
    accu = Accuracy().cuda()

    target_model_name = args.model_A
    aux_model_name = args.model_B
    target_model = all_classifiers[target_model_name](pretrained=True).eval().cuda()
    aux_model = all_classifiers[aux_model_name](pretrained=True).eval().cuda()
    args.model_name = target_model_name + '-X-' + aux_model_name + '.ptl'
    print(args.model_name)

    test = C10IMGDATA(args).dataloader()
    test = iter(test)
    target_model_accuracies = []
    aux_model_accuracies = []
    ssims = []

    result_file = r'mobile_result_mmsys2022.txt'

    for i in range(int(5000/args.batch_size)):
        original, perturbed, label = next(test)
        original = original.cuda()
        perturbed = perturbed.cuda()
        label = label.cuda()

        target_model_predict = target_model(perturbed)
        target_model_predict = m(target_model_predict)
        aux_model_predict = aux_model(perturbed)
        aux_model_predict = m(aux_model_predict)

        target_model_accuracy = accu(target_model_predict, label).item()
        target_model_accuracies.append(target_model_accuracy)
        aux_model_accuracy = accu(aux_model_predict, label).item()
        aux_model_accuracies.append(aux_model_accuracy)
        ssim = ssim_calc(unnormalize(original), unnormalize(perturbed)).item()
        ssims.append(ssim)

    ave_target_model_accuracy = sum(target_model_accuracies) / len(target_model_accuracies)
    ave_aux_model_accuracy = sum(aux_model_accuracies) / len(aux_model_accuracies)
    ave_ssim = sum(ssims) / len(ssims)

    print(ave_target_model_accuracy)
    print(ave_aux_model_accuracy)
    print(ave_ssim)

    with open(result_file, 'a+') as f:
        f.write(args.model_name+'\n')
        f.write(str(ave_target_model_accuracy)+'\n')
        f.write(str(ave_aux_model_accuracy)+'\n')
        f.write(str(ave_ssim)+'\n')
        f.write('\n')


