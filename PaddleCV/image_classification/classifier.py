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

def merge_cfg(params):
    args = parse_args()
    for k, v in params.items():
        if hasattr(args, k):
            setattr(args, k, v)
    return args


class Classifier(object):
    def __init__(self,
                 work_dir,
                 model_name,
                 num_classes=None,
                 use_pretrained_weights=False,
                 use_gpu=True):
        self.model_name = model_name
        self.work_dir = work_dir
        self.use_pretrained_weights = use_pretrained_weights
        if num_classes is None:
            raise Exception("num_classes must be defined in Classifier")
        self.class_dim = num_classes
        self.use_gpu = use_gpu
        self.exe = None
        if use_pretrained_weights:
            pretrain_dir = path_join(work_dir, "pretrain")
            self.pretrained_weights_dir = get_pretrained_weights(model_name,
                                                                 pretrain_dir)

    def fit(self, data_dir, num_epochs=120, lr=0.1, batch_size=8):
        cfg = merge_cfg(locals())
        cfg.class_dim = self.class_dim
        cfg.test_batch_size = batch_size
        self.model_save_dir = path_join(self.work_dir, "saved_model")
        if self.use_pretrained_weights:
            cfg.pretrained_model = self.pretrained_weights_dir
            remove_fc_vars(self.pretrained_weights_dir, self.model_name)
        cfg.model = self.model_name
        cfg.use_gpu = self.use_gpu
        cfg.model_save_dir = osp.join(self.work_dir, 'output')
        train_txt_path = osp.join(data_dir, 'train_list.txt')
        assert osp.exists(train_txt_path), \
            'The train list file ({}) is not existed!'.format(train_txt_path)
        with open(train_txt_path, "r") as flist:
            full_lines = [line.strip() for line in flist]
            cfg.total_images = len(full_lines)
        check_args(cfg)
        self.cfg = cfg
        self.train_res = train(cfg)

    def predict(self, img_file):
        self.cfg.img_file = img_file
        check_gpu()
        check_version()
        infer(self.cfg, self.train_res[0], self.train_res[1], self.train_res[2],
              self.train_res[4])

    def eval(self, data_dir):
        self.cfg.data_dir = data_dir
        check_gpu()
        check_version()
        eval(self.cfg, self.train_res[0], self.train_res[1], self.train_res[2],
             self.train_res[3])

    def load_model(self, model_dir):
        assert os.path.exists(model_dir), \
            'The model path {} is not exist!'.format(model_dir)
        import reader 
        cfg = merge_cfg(locals())
        cfg.class_dim = self.class_dim
        cfg.model = self.model_name
        cfg.use_gpu = self.use_gpu
        cfg.total_images = 1
        cfg.batch_size = 1
        check_args(cfg)
        self.cfg = cfg
        startup_prog = fluid.Program()
        test_prog = fluid.Program()
        test_out, res_out = build_program(
            is_train=False,
            main_prog=test_prog,
            startup_prog=startup_prog,
            args=self.cfg)
        test_data_loader = test_out[-1]
        test_fetch_vars = test_out[:-1]
        test_fetch_list = [var.name for var in test_fetch_vars]
        test_prog = test_prog.clone(for_test=True)
        place = fluid.CUDAPlace(0) if self.cfg.use_gpu else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_prog)
        fluid.io.load_persistables(exe, model_dir)
        self.train_res = [startup_prog, test_prog, test_data_loader, test_fetch_list, res_out]

madel_path = '/all/image_classification/output/ResNet18/'
mymodel = Classifier(work_dir="myproject", model_name="ResNet18", num_classes=283)
mymodel.load_model(madel_path)
print('______________infer-------------------')
mymodel.predict('/all/mini_data/kmeans_data/0/9.jpg')
