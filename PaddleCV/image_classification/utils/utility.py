# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import distutils.util
import numpy as np
import six
import argparse
import functools
import logging
import sys
import os
import warnings
import signal
import easydict

import paddle
import paddle.fluid as fluid
from paddle.fluid.wrapped_decorator import signature_safe_contextmanager
from paddle.fluid.framework import Program, program_guard, name_scope, default_main_program
from paddle.fluid import unique_name, layers
from utils import dist_utils


def parse_args():
    """Add arguments

    Returns: 
        all training args
    """
    args = easydict.EasyDict({
        "use_gpu": True,
        "model_save_dir": "./output",
        "data_dir": "./data/ILSVRC2012/",
        "pretrained_model": None,
        "checkpoint": None,
        "print_step": 10,
        "save_step": 1,
        "model": "ResNet50",
        "total_images": 1281167,
        "num_epochs": 120,
        "class_dim": 1000,
        "image_shape": "3,224,224",
        "batch_size": 8,
        "test_batch_size": 16,
        "lr": 0.1,
        "lr_strategy": "piecewise_decay",
        "l2_decay": 1e-04,
        "momentum_rate": 0.9,
        "warm_up_epochs": 5.0,
        "decay_epochs": 2.4,
        "decay_rate": 0.97,
        "drop_connect_rate": 0.2,
        "step_epochs": [30, 60, 90],
        "lower_scale": 0.08,
        "lower_ratio": 3. / 4,
        "upper_ratio": 4. / 3,
        "resize_short_size": 256,
        "crop_size": 224,
        "use_mixup": False,
        "mixup_alpha": 0.2,
        "reader_thread": 8,
        "reader_buf_size": 2048,
        "interpolation": None,
        "use_aa": False,
        "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225],
        "use_label_smoothing": False,
        "label_smoothing_epsilon": 0.1,
        "random_seed": None,
        "use_ema": False,
        "ema_decay": 0.9999,
        "padding_type": "SAME",
        "use_se": True,
        "img_file": "",
        "topk": 1,
    })
    return args


def check_gpu():
    """   
    Log error and exit when set use_gpu=true in paddlepaddle
    cpu ver sion.
    """
    logger = logging.getLogger(__name__)
    err = "Config use_gpu cannot be set as true while you are " \
                "using paddlepaddle cpu version ! \nPlease try: \n" \
                "\t1. Install paddlepaddle-gpu to run model on GPU \n" \
                "\t2. Set use_gpu as false in config file to run " \
                "model on CPU"

    try:
        if args.use_gpu and not fluid.is_compiled_with_cuda():
            print(err)
            sys.exit(1)
    except Exception as e:
        pass


def check_version():
    """
    Log error and exit when the installed version of paddlepaddle is
    not satisfied.
    """
    err = "PaddlePaddle version 1.6 or higher is required, " \
          "or a suitable develop version is satisfied as well. \n" \
          "Please make sure the version is good with your code." \

    try:
        fluid.require_version('1.6.0')
    except Exception as e:
        print(err)
        sys.exit(1)


def check_args(args):
    """check arguments before running

    Args:
        all arguments
    """

    # check models name
    sys.path.append("..")
    import models
    model_list = [m for m in dir(models) if "__" not in m]
    assert args.model in model_list, "{} is not in lists: {}, please check the model name".format(
        args.model, model_list)

    # check learning rate strategy
    lr_strategy_list = [
        "piecewise_decay", "cosine_decay", "linear_decay",
        "cosine_decay_warmup", "exponential_decay_warmup"
    ]
    if args.lr_strategy not in lr_strategy_list:
        warnings.warn(
            "\n{} is not in lists: {}, \nUse default learning strategy now.".
            format(args.lr_strategy, lr_strategy_list))
        args.lr_strategy = "default_decay"
    # check confict of GoogLeNet and mixup
    if args.model == "GoogLeNet":
        assert args.use_mixup == False, "Cannot use mixup processing in GoogLeNet, please set use_mixup = False."

    if args.interpolation:
        assert args.interpolation in [
            0, 1, 2, 3, 4
        ], "Wrong interpolation, please set:\n0: cv2.INTER_NEAREST\n1: cv2.INTER_LINEAR\n2: cv2.INTER_CUBIC\n3: cv2.INTER_AREA\n4: cv2.INTER_LANCZOS4"

    if args.padding_type:
        assert args.padding_type in [
            "SAME", "VALID", "DYNAMIC"
        ], "Wrong padding_type, please set:\nSAME\nVALID\nDYNAMIC"

    assert args.checkpoint is None or args.pretrained_model is None, "Do not init model by checkpoint and pretrained_model both."

    # check pretrained_model path for loading
    if args.pretrained_model is not None:
        assert isinstance(args.pretrained_model, str)
        assert os.path.isdir(
            args.
            pretrained_model), "please support available pretrained_model path."

    #FIXME: check checkpoint path for saving
    if args.checkpoint is not None:
        assert isinstance(args.checkpoint, str)
        assert os.path.isdir(
            args.checkpoint
        ), "please support available checkpoint path for initing model."

    # check params for loading
    """
    if args.save_params:
        assert isinstance(args.save_params, str)
        assert os.path.isdir(
            args.save_params), "please support available save_params path."
    """

    # check gpu: when using gpu, the number of visible cards should divide batch size
    if args.use_gpu:
        assert args.batch_size % fluid.core.get_cuda_device_count(
        ) == 0, "please support correct batch_size({}), which can be divided by available cards({}), you can change the number of cards by indicating: export CUDA_VISIBLE_DEVICES= ".format(
            args.batch_size, fluid.core.get_cuda_device_count())

    # check data directory
    assert os.path.isdir(
        args.data_dir
    ), "Data doesn't exist in {}, please load right path".format(args.data_dir)

    #check gpu

    check_gpu()
    check_version()


def init_model(exe, args, program):
    if args.checkpoint:
        fluid.io.load_persistables(exe, args.checkpoint, main_program=program)
        print("Finish initing model from %s" % (args.checkpoint))

    if args.pretrained_model:

        def if_exist(var):
            return os.path.exists(os.path.join(args.pretrained_model, var.name))

        fluid.io.load_vars(
            exe,
            args.pretrained_model,
            main_program=program,
            predicate=if_exist)


def save_model(args, exe, train_prog, info):
    model_path = os.path.join(args.model_save_dir, args.model, str(info))
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    fluid.io.save_persistables(exe, model_path, main_program=train_prog)
    print("Already save model in %s" % (model_path))


def create_data_loader(is_train, args):
    """create data_loader

    Usage:
        Using mixup process in training, it will return 5 results, include data_loader, image, y_a(label), y_b(label) and lamda, or it will return 3 results, include data_loader, image, and label.

    Args: 
        is_train: mode
        args: arguments

    Returns:
        data_loader and the input data of net, 
    """
    image_shape = [int(m) for m in args.image_shape.split(",")]

    feed_image = fluid.data(
        name="feed_image",
        shape=[None] + image_shape,
        dtype="float32",
        lod_level=0)

    feed_label = fluid.data(
        name="feed_label", shape=[None, 1], dtype="int64", lod_level=0)
    feed_y_a = fluid.data(
        name="feed_y_a", shape=[None, 1], dtype="int64", lod_level=0)

    if is_train and args.use_mixup:
        feed_y_b = fluid.data(
            name="feed_y_b", shape=[None, 1], dtype="int64", lod_level=0)
        feed_lam = fluid.data(
            name="feed_lam", shape=[None, 1], dtype="float32", lod_level=0)

        data_loader = fluid.io.DataLoader.from_generator(
            feed_list=[feed_image, feed_y_a, feed_y_b, feed_lam],
            capacity=64,
            use_double_buffer=True,
            iterable=False)
        return data_loader, [feed_image, feed_y_a, feed_y_b, feed_lam]
    else:
        data_loader = fluid.io.DataLoader.from_generator(
            feed_list=[feed_image, feed_label],
            capacity=64,
            use_double_buffer=True,
            iterable=False)

        return data_loader, [feed_image, feed_label]


def print_info(pass_id, batch_id, print_step, metrics, time_info, info_mode):
    """print function

    Args:
        pass_id: epoch index
        batch_id: batch index
        print_step: the print_step arguments
        metrics: message to print
        time_info: time infomation
        info_mode: mode
    """
    if info_mode == "batch":
        if batch_id % print_step == 0:
            #if isinstance(metrics,np.ndarray):
            # train and mixup output
            if len(metrics) == 2:
                loss, lr = metrics
                print(
                    "[Pass {0}, train batch {1}] \tloss {2}, lr {3}, elapse {4}".
                    format(pass_id, batch_id, "%.5f" % loss, "%.5f" % lr,
                           "%2.4f sec" % time_info))
            # train and no mixup output
            elif len(metrics) == 4:
                loss, acc1, acc5, lr = metrics
                print(
                    "[Pass {0}, train batch {1}] \tloss {2}, acc1 {3}, acc5 {4}, lr {5}, elapse {6}".
                    format(pass_id, batch_id, "%.5f" % loss, "%.5f" % acc1,
                           "%.5f" % acc5, "%.5f" % lr, "%2.4f sec" % time_info))
            # test output
            elif len(metrics) == 3:
                loss, acc1, acc5 = metrics
                print(
                    "[Pass {0}, test  batch {1}] \tloss {2}, acc1 {3}, acc5 {4}, elapse {5}".
                    format(pass_id, batch_id, "%.5f" % loss, "%.5f" % acc1,
                           "%.5f" % acc5, "%2.4f sec" % time_info))
            else:
                raise Exception(
                    "length of metrics {} is not implemented, It maybe caused by wrong format of build_program_output".
                    format(len(metrics)))
            sys.stdout.flush()

    elif info_mode == "epoch":
        ## TODO add time elapse
        #if isinstance(metrics,np.ndarray):
        if len(metrics) == 5:
            train_loss, _, test_loss, test_acc1, test_acc5 = metrics
            print(
                "[End pass {0}]\ttrain_loss {1}, test_loss {2}, test_acc1 {3}, test_acc5 {4}".
                format(pass_id, "%.5f" % train_loss, "%.5f" % test_loss, "%.5f"
                       % test_acc1, "%.5f" % test_acc5))
        elif len(metrics) == 7:
            train_loss, train_acc1, train_acc5, _, test_loss, test_acc1, test_acc5 = metrics
            print(
                "[End pass {0}]\ttrain_loss {1}, train_acc1 {2}, train_acc5 {3},test_loss {4}, test_acc1 {5}, test_acc5 {6}".
                format(pass_id, "%.5f" % train_loss, "%.5f" % train_acc1, "%.5f"
                       % train_acc5, "%.5f" % test_loss, "%.5f" % test_acc1,
                       "%.5f" % test_acc5))
        sys.stdout.flush()
    elif info_mode == "ce":
        raise Warning("CE code is not ready")
    else:
        raise Exception("Illegal info_mode")


def best_strategy_compiled(args, program, loss, exe):
    """make a program which wrapped by a compiled program
    """

    if os.getenv('FLAGS_use_ngraph'):
        return program
    else:
        build_strategy = fluid.compiler.BuildStrategy()
        #Feature will be supported in Fluid v1.6
        #build_strategy.enable_inplace = True

        exec_strategy = fluid.ExecutionStrategy()
        exec_strategy.num_threads = fluid.core.get_cuda_device_count()
        exec_strategy.num_iteration_per_drop_scope = 10

        num_trainers = int(os.environ.get('PADDLE_TRAINERS_NUM', 1))
        if num_trainers > 1 and args.use_gpu:
            dist_utils.prepare_for_multi_process(exe, build_strategy, program)
            # NOTE: the process is fast when num_threads is 1
            # for multi-process training.
            exec_strategy.num_threads = 1

        compiled_program = fluid.CompiledProgram(program).with_data_parallel(
            loss_name=loss.name,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy)

        return compiled_program


class ExponentialMovingAverage(object):
    def __init__(self,
                 decay=0.999,
                 thres_steps=None,
                 zero_debias=False,
                 name=None):
        self._decay = decay
        self._thres_steps = thres_steps
        self._name = name if name is not None else ''
        self._decay_var = self._get_ema_decay()

        self._params_tmps = []
        for param in default_main_program().global_block().all_parameters():
            if param.do_model_average != False:
                tmp = param.block.create_var(
                    name=unique_name.generate(".".join(
                        [self._name + param.name, 'ema_tmp'])),
                    dtype=param.dtype,
                    persistable=False,
                    stop_gradient=True)
                self._params_tmps.append((param, tmp))

        self._ema_vars = {}
        for param, tmp in self._params_tmps:
            with param.block.program._optimized_guard(
                [param, tmp]), name_scope('moving_average'):
                self._ema_vars[param.name] = self._create_ema_vars(param)

        self.apply_program = Program()
        block = self.apply_program.global_block()
        with program_guard(main_program=self.apply_program):
            decay_pow = self._get_decay_pow(block)
            for param, tmp in self._params_tmps:
                param = block._clone_variable(param)
                tmp = block._clone_variable(tmp)
                ema = block._clone_variable(self._ema_vars[param.name])
                layers.assign(input=param, output=tmp)
                # bias correction
                if zero_debias:
                    ema = ema / (1.0 - decay_pow)
                layers.assign(input=ema, output=param)

        self.restore_program = Program()
        block = self.restore_program.global_block()
        with program_guard(main_program=self.restore_program):
            for param, tmp in self._params_tmps:
                tmp = block._clone_variable(tmp)
                param = block._clone_variable(param)
                layers.assign(input=tmp, output=param)

    def _get_ema_decay(self):
        with default_main_program()._lr_schedule_guard():
            decay_var = layers.tensor.create_global_var(
                shape=[1],
                value=self._decay,
                dtype='float32',
                persistable=True,
                name="scheduled_ema_decay_rate")

            if self._thres_steps is not None:
                decay_t = (self._thres_steps + 1.0) / (self._thres_steps + 10.0)
                with layers.control_flow.Switch() as switch:
                    with switch.case(decay_t < self._decay):
                        layers.tensor.assign(decay_t, decay_var)
                    with switch.default():
                        layers.tensor.assign(
                            np.array(
                                [self._decay], dtype=np.float32),
                            decay_var)
        return decay_var

    def _get_decay_pow(self, block):
        global_steps = layers.learning_rate_scheduler._decay_step_counter()
        decay_var = block._clone_variable(self._decay_var)
        decay_pow_acc = layers.elementwise_pow(decay_var, global_steps + 1)
        return decay_pow_acc

    def _create_ema_vars(self, param):
        param_ema = layers.create_global_var(
            name=unique_name.generate(self._name + param.name + '_ema'),
            shape=param.shape,
            value=0.0,
            dtype=param.dtype,
            persistable=True)

        return param_ema

    def update(self):
        """
        Update Exponential Moving Average. Should only call this method in
        train program.
        """
        param_master_emas = []
        for param, tmp in self._params_tmps:
            with param.block.program._optimized_guard(
                [param, tmp]), name_scope('moving_average'):
                param_ema = self._ema_vars[param.name]
                if param.name + '.master' in self._ema_vars:
                    master_ema = self._ema_vars[param.name + '.master']
                    param_master_emas.append([param_ema, master_ema])
                else:
                    ema_t = param_ema * self._decay_var + param * (
                        1 - self._decay_var)
                    layers.assign(input=ema_t, output=param_ema)

        # for fp16 params
        for param_ema, master_ema in param_master_emas:
            default_main_program().global_block().append_op(
                type="cast",
                inputs={"X": master_ema},
                outputs={"Out": param_ema},
                attrs={
                    "in_dtype": master_ema.dtype,
                    "out_dtype": param_ema.dtype
                })

    @signature_safe_contextmanager
    def apply(self, executor, need_restore=True):
        """
        Apply moving average to parameters for evaluation.

        Args:
            executor (Executor): The Executor to execute applying.
            need_restore (bool): Whether to restore parameters after applying.
        """
        executor.run(self.apply_program)
        try:
            yield
        finally:
            if need_restore:
                self.restore(executor)

    def restore(self, executor):
        """Restore parameters.

        Args:
            executor (Executor): The Executor to execute restoring.
        """
        executor.run(self.restore_program)
