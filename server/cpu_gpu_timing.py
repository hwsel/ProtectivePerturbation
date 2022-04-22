import os
from argparse import ArgumentParser
import timeit

from pytorch_lightning import seed_everything

from data import CIFAR10Data
from module import AttackModule
from module_noModelB import AttackModule as AttackModule_noB
from utils.all_classifiers import classifier_list, all_classifiers


def get_running_time(model, dataset, isGPU):
    cnt = 0
    total_time = 0
    total_size = 0

    if isGPU:
        model = model.cuda()

    dataloader = dataset.test_dataloader()
    dataloader = iter(dataloader)

    while cnt < int(5000 / args.batch_size):
        data, label = next(dataloader)
        if isGPU:
            data = data.cuda()
        t0 = timeit.default_timer()
        model(data)
        t1 = timeit.default_timer()
        total_size = total_size + len(data)
        total_time = total_time + t1 - t0
        cnt = cnt + 1

    return total_time / total_size * 1000.  # return time in ms


def main(args):
    seed_everything(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    generator_cpu_time_ave = []
    generator_gpu_time_ave = []
    dnn_cpu_time = []
    dnn_gpu_time = []
    dataset = CIFAR10Data(args)
    for model_A in classifier_list:
        print('current model A', model_A)
        generator_cpu_time = []
        generator_gpu_time = []
        for model_B in classifier_list:
            print('current model B', model_B)
            model_path = os.path.join(args.model_dir, model_A+'-X-'+model_B+'.ckpt')
            if model_A == model_B:
                model = AttackModule_noB.load_from_checkpoint(model_path).attack_model.eval()
            else:
                model = AttackModule.load_from_checkpoint(model_path).attack_model.eval()

            generator_cpu_time.append(get_running_time(model, dataset, False))
            generator_gpu_time.append(get_running_time(model, dataset, True))

        generator_cpu_time_ave.append(sum(generator_cpu_time) / len(generator_cpu_time))
        generator_gpu_time_ave.append(sum(generator_gpu_time) / len(generator_gpu_time))

        dnn = all_classifiers[model_A](pretrained=True).eval()
        dnn_cpu_time.append(get_running_time(dnn, dataset, False))
        dnn_gpu_time.append(get_running_time(dnn, dataset, True))

    print(classifier_list)
    print(generator_cpu_time_ave)
    print(generator_gpu_time_ave)
    print(dnn_cpu_time)
    print(dnn_gpu_time)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="/home/tangbao/data/cifar10")
    parser.add_argument("--model_dir", type=str, default="./generator_models")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gpu_id", type=str, default="0")
    args = parser.parse_args()

    main(args)
