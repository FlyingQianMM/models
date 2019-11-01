import os
from imagenet_pretrained_weights import *
from utils import *
from train import *
from eval import *
from infer import *
import paddle.fluid as fluid


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
    def __init__(self,
                 work_dir,
                 model_name,
                 use_pretrained_weights=False,
                 use_gpu=True):
        self.model_name = model_name
        self.work_dir = work_dir
        self.use_pretrained_weights = use_pretrained_weights
        self.use_gpu = use_gpu
        self.exe = None
        if use_pretrained_weights:
            pretrain_dir = path_join(work_dir, "pretrain")
            self.pretrained_weights_dir = get_pretrained_weights(model_name,
                                                                 pretrain_dir)

    def fit(self, data_dir, num_epochs=120, lr=0.1, class_dim=None):
        cfg = merge_cfg(**locals())
        self.model_save_dir = path_join(self.work_dir, "saved_model")
        if self.use_pretrained_weights:
            cfg.pretrained_model = self.pretrained_weights_dir
            remove_fc_vars(self.pretrained_weights_dir, self.model_name)
        cfg.model = self.model_name
        cfg.use_gpu = self.use_gpu
        cfg.class_dim = class_dim
        self.cfg = cfg
        self.class_dim = class_dim
        cfg.model_save_dir = osp.join(self.work_dir, 'output')
        train_txt_path = osp.join(data_dir, 'train_list.txt')
        assert osp.exists(train_txt_path), \
            'The train list file ({}) is not existed!'.format(train_txt_path)
        with open(train_txt_path, "r") as flist:
            full_lines = [line.strip() for line in flist]
            cfg.total_images = len(full_lines)
        check_args(cfg)
        self.train_res = train(cfg)

    def predict(self, img_file):
        cfg = merge_cfg(**locals())
        cfg.model = self.model_name
        cfg.use_gpu = self.use_gpu
        cfg.class_dim = self.class_dim
        check_gpu()
        check_version()
        infer(cfg, self.train_res[0], self.train_res[1], self.train_res[2],
              self.train_res[4])

    def eval(self, data_dir):
        cfg = merge_cfg(**locals())
        cfg.model = self.model_name
        cfg.use_gpu = self.use_gpu
        cfg.class_dim = self.class_dim
        check_gpu()
        check_version()
        eval(cfg, self.train_res[0], self.train_res[1], self.train_res[2],
             self.train_res[3])

    def load_model(
            self,
            model_dir, ):
        if self.exe is None:
            place = fluid.CUDAPlace(0) if self.use_gpu else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            self.exe = exe
        fluid.io.load_persistables(self.exe, model_dir)


#    def save_inference_model(self):
mymodel = Classifier(
    work_dir="myproject", model_name="ResNet18", use_pretrained_weights=True)
print(mymodel.pretrained_weights_dir)
mymodel.fit(data_dir="../mini_data", num_epochs=1, lr=0.1, class_dim=283)
print('________________________________eval__________________________')
mymodel.eval(data_dir="../mini_data")
print('________________________________infer__________________________')
mymodel.predict(img_file='/all/mini_data/kmeans_data/0/9.jpg')
