
import os

import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

from module import AttackModule
from module_noModelB import AttackModule as AttackModule_noB
from utils.all_classifiers import all_classifiers, classifier_list


CKPT_FOLDER = './generator_models'
GENERATOR_MODEL_SAVE_FOLDER = './mobile_model/generator'
TARGET_MODEL_SAVE_FOLDER = './mobile_model/target_model'


def save_mobile_model(model, full_path):
    example = torch.rand(1, 3, 32, 32)
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module_optimized = optimize_for_mobile(traced_script_module)
    traced_script_module_optimized._save_for_lite_interpreter(full_path)


def export_mobile_generator_model():
    model_list = os.listdir(CKPT_FOLDER)
    model_list.remove('PUT_original_model_HERE')

    for model in model_list:
        print('exporting', model)
        ckpt_path = os.path.join(CKPT_FOLDER, model)
        submodels = model[:-5].split('-X-')
        if submodels[0] == submodels[1]:
            module = AttackModule_noB.load_from_checkpoint(ckpt_path).eval()
        else:
            module = AttackModule.load_from_checkpoint(ckpt_path).eval()
        unet = module.attack_model.eval()
        save_mobile_model(unet, os.path.join(GENERATOR_MODEL_SAVE_FOLDER, model[:-5] + ".ptl"))


def export_mobile_targetmodel():
    for classifier in classifier_list:
        print('exporting', classifier)
        model = all_classifiers[classifier](pretrained=True).eval()
        save_mobile_model(model, os.path.join(TARGET_MODEL_SAVE_FOLDER, classifier + ".ptl"))


if __name__ == '__main__':
    export_mobile_generator_model()
    export_mobile_targetmodel()
