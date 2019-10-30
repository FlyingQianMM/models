import os
from imagenet_pretrained_weights import *
from utils import *
from train import *

def path_join(dir_path, file_or_dir):
    res = os.path.join(dir_path, file_or_dir)
    return os.path.normpath(res)

def merge_cfg(**kwargs):
    args = parse_args()
    for k, v in kwargs.items():
        print("=====", k, v)
        if hasattr(args, k):
            setattr(args, k, v)
    return args

class Classifier(object):
    def __init__(self, work_dir, model_name, use_pretrained_weights=False):
        self.model_name = model_name
        self.work_dir = work_dir
        self.use_pretrained_weights = use_pretrained_weights
        if use_pretrained_weights:
            pretrain_dir = path_join(work_dir, "pretrain")
            self.pretrained_weights_dir = get_pretrained_weights(model_name, pretrain_dir)

    def fit(self, data_dir, num_epochs=120, lr=0.1, class_dim=None, total_images=None):
        cfg = merge_cfg(**locals())
        self.model_save_dir = path_join(self.work_dir, "saved_model")
        if self.use_pretrained_weights:
            cfg.pretrained_model = self.pretrained_weights_dir
            remove_fc_vars(self.pretrained_weights_dir, self.model_name)
        cfg.model = self.model_name
        check_args(cfg)
        train(cfg)

#    def predict(self, img_file):
#        pass
#
#    def eval(self, val_data):
#        pass
#
#    def load_model(self):
#        pass
#
#    def save_inference_model(self):
mymodel = Classifier(work_dir="myproject", model_name="ResNet18", use_pretrained_weights=True)
print(mymodel.pretrained_weights_dir)
mymodel.fit(data_dir="/ssd3/jiangxiaobai/data", num_epochs=2, lr=0.1, class_dim=283, total_images=2830)
