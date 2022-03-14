#!/usr/bin/env bash
for MODEL_A in vgg13_bn vgg16_bn resnet18 resnet34 densenet121 mobilenet_v2 googlenet inception_v3
do
	for MODEL_B in vgg13_bn vgg16_bn resnet18 resnet34 densenet121 mobilenet_v2 googlenet inception_v3
	do
        	python train.py --model_A $MODEL_A --model_B $MODEL_B --gpu_id $1 --ssim_weight 0.5 --description "$MODEL_A-X-$MODEL_B"
	done
done

