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
import reader
import models
from utils import *


def inference(args, test_data_loader, exe, test_prog, test_fetch_list):
    test_batch_time_record = []
    test_batch_metrics_record = []
    test_batch_id = 0
    test_data_loader.start()
    try:
        while True:
            t1 = time.time()
            result = exe.run(program=test_prog, fetch_list=test_fetch_list)
            t2 = time.time()
            test_batch_elapse = t2 - t1
            result = result[0][0]
            pred_label = np.argsort(result)[::-1][:args.topk]
            print(pred_label)

    except fluid.core.EOFException:
        test_data_loader.reset()


def infer(args, startup_program, test_program, test_data_loader,
          test_fetch_list):
    gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
    place = fluid.CUDAPlace(gpu_id) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    num_trainers = int(os.environ.get('PADDLE_TRAINERS_NUM', 1))
    imagenet_reader = reader.ImageNetReader(0 if num_trainers > 1 else None)
    test_reader = imagenet_reader.test(settings=args)
    test_data_loader.set_sample_list_generator(test_reader, place)
    inference(args, test_data_loader, exe, test_program, test_fetch_list)
