from utils.all_classifiers import all_classifiers
from pytorch_lightning.callbacks import Callback


class AttackCallback(Callback):
    def on_test_start(self, trainer, module):
        module.model_A = all_classifiers[module.hparams.model_A](pretrained=True).eval()
        module.model_B = all_classifiers[module.hparams.model_B](pretrained=True).eval()
        module.model_A.to(module.device)
        module.model_B.to(module.device)
