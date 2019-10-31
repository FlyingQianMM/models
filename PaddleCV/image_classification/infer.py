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



def infer(args, startup_program, test_program):
    image_shape = [int(m) for m in args.image_shape.split(",")]
    model_list = [m for m in dir(models) if "__" not in m]
    assert args.model in model_list, "{} is not in lists: {}".format(args.model,
                                                                     model_list)
    image = fluid.data(
        name='image', shape=[None] + image_shape, dtype='float32')

    if args.model.startswith('EfficientNet'):
        model = models.__dict__[args.model](is_test=True,
                                            padding_type=args.padding_type,
                                            use_se=args.use_se)
    else:
        model = models.__dict__[args.model]()

    if args.model == "GoogLeNet":
        out, _, _ = model.net(input=image, class_dim=args.class_dim)
    else:
        out = model.net(input=image, class_dim=args.class_dim)
        out = fluid.layers.softmax(out)


    fetch_list = [out.name]

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_program)


    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    args.test_batch_size = 1
    imagenet_reader = reader.ImageNetReader()
    test_reader = imagenet_reader.test(settings=args)
    feeder = fluid.DataFeeder(place=place, feed_list=[image])

    TOPK = args.topk
    assert os.path.exists(args.label_path), "Index file doesn't exist!"
    f = open(args.label_path)
    label_dict = {}
    for item in f.readlines():
        key = item.split(" ")[0]
        value = [l.replace("\n", "") for l in item.split(" ")[1:]]
        label_dict[key] = value

#     for block in test_program.blocks:
#         for param in block.all_parameters():
#             pd_var = fluid.global_scope().find_var(param.name)
#             pd_param = pd_var.get_tensor()
#             pd_param.set(weights[param.name], place)
            
    
    for batch_id, data in enumerate(test_reader()):
        result = exe.run(test_program,
                         fetch_list=fetch_list,
                         feed=feeder.feed(data))
        result = result[0][0]
        pred_label = np.argsort(result)[::-1][:TOPK]
        print(pred_label)
        '''
        readable_pred_label = []
        for label in pred_label:
            readable_pred_label.append(label_dict[str(label)])
        print("Test-{0}-score: {1}, class{2} {3}".format(batch_id, result[
            pred_label], pred_label, readable_pred_label))
        sys.stdout.flush()
        '''

    print(np.array(fluid.global_scope().find_var('bn2a_branch1_mean').get_tensor()))