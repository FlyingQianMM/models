#copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

import os
import time
import sys
import math
import numpy as np
import argparse
import functools

import paddle
import paddle.fluid as fluid
from image_classification import reader
from image_classification import models
from image_classification.utils import *
from image_classification.build_model import create_model


def validate(args, test_data_loader, exe, test_prog, test_fetch_list):
    test_batch_time_record = []
    test_batch_metrics_record = []
    test_batch_id = 0
    test_data_loader.start()
    try:
        while True:
            t1 = time.time()
            test_batch_metrics = exe.run(program=test_prog,
                                         fetch_list=test_fetch_list)
            t2 = time.time()
            test_batch_elapse = t2 - t1
            test_batch_time_record.append(test_batch_elapse)

            test_batch_metrics_avg = np.mean(
                np.array(test_batch_metrics), axis=1)
            test_batch_metrics_record.append(test_batch_metrics_avg)

            print_info("eval", test_batch_id, args.print_step,
                       test_batch_metrics_avg, test_batch_elapse, "batch")
            sys.stdout.flush()
            test_batch_id += 1

    except fluid.core.EOFException:
        test_data_loader.reset()

    test_epoch_time_avg = np.mean(np.array(test_batch_time_record))
    test_epoch_metrics_avg = np.mean(
        np.array(test_batch_metrics_record), axis=0)
    loss, acc1, acc5 = list(test_epoch_metrics_avg)
    print("Eval results: eval_loss {}, eval_acc1 {}, eval_acc5 {}".format( \
            str(loss), str(acc1), str(acc5)))


def eval(args, startup_program, test_program, test_data_loader,
         test_fetch_list):
    gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
    place = fluid.CUDAPlace(gpu_id) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    num_trainers = int(os.environ.get('PADDLE_TRAINERS_NUM', 1))
    imagenet_reader = reader.ImageNetReader(0 if num_trainers > 1 else None)
    test_reader = imagenet_reader.val(settings=args)
    test_data_loader.set_sample_list_generator(test_reader, place)
    validate(args, test_data_loader, exe, test_program, test_fetch_list)
