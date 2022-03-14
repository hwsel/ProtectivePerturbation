#!/usr/bin/env bash
for MODEL_A in vgg13_bn vgg16_bn resnet18 resnet34 densenet121 mobilenet_v2 googlenet inception_v3
do
	for MODEL_B in vgg13_bn vgg16_bn resnet18 resnet34 densenet121 mobilenet_v2 googlenet inception_v3
	do
        	python mobile_result_file_gen.py --model_A $MODEL_A --model_B $MODEL_B
	done
done

