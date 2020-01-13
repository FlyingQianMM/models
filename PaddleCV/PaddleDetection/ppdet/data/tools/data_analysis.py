#!/usr/bin/env python
# coding: utf-8
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import glob
import json
import os
import os.path as osp
import sys
import shutil

import numpy as np
import cv2 as cv
import xml.etree.ElementTree as ET

import PIL.ImageDraw


def voc_primary_analysis(data_dir):
    formatcorrect = True
    if not osp.exists(osp.join(data_dir, 'trainval.txt')):
        print(
            "[VOC check ] The file 'trainval.txt' does not exist in {}".format(
                data_dir))
        formatcorrect = False
    if not osp.exists(osp.join(data_dir, 'test.txt')):
        print("[VOC check ] The file 'test.txt' does not exist in {}".format(
            data_dir))
        formatcorrect = False
    if not osp.exists(osp.join(data_dir, 'VOCdevkit')):
        print(
            "[VOC check ] The folder 'VOCdevkit' does not exist in {}".format(
                data_dir))
        formatcorrect = False

    return formatcorrect


def voc_intermediate_analysis(data_dir):
    anno_paths = [osp.join(data_dir, 'trainval.txt'), \
                 osp.join(data_dir, 'test.txt')]
    for anno_path in anno_paths:
        records = 0
        formatcorrect = True
        with open(anno_path, 'r') as fr:
            while True:
                line = fr.readline()
                if not line:
                    break
                img_file, xml_file = [os.path.join(data_dir, x) \
                        for x in line.strip().split()[:2]]
                if not os.path.isfile(img_file):
                    print("[VOC check ] The file {} does not exist!".format(
                        img_file))
                    continue
                if not os.path.isfile(xml_file):
                    print("[VOC check ] The file {} does not exist!".format(
                        xml_file))
                    continue

                try:
                    image = cv.imread(img_file)
                    shape = image.shape
                except:
                    print('[VOC check ] Something wrong in the file {}'.format(
                        img_file))
                    continue

                tree = ET.parse(xml_file)
                objs = tree.findall('object')
                if len(objs) == 0:
                    print('[VOC check ] Not find any object in {}'.format(
                        xml_file))
                if tree.find('size') is None or tree.find('size').find('width') is None \
                    or tree.find('size').find('height') is None:
                    print(
                        '[VOC check ] size information is not correctly set in {}'
                        .format(xml_file))
                    continue
                im_w = float(tree.find('size').find('width').text)
                im_h = float(tree.find('size').find('height').text)
                valid_objs_num = len(objs)
                for i, obj in enumerate(objs):
                    if obj.find('name') is None:
                        print(
                            '[VOC check ] name of {}-th object does not exist in {}'
                            .format(i + 1, xml_file))
                        valid_objs_num -= 1
                        continue
                    if obj.find('difficult') is None:
                        print(
                            '[VOC check ] difficult of {}-th object does not exist in {}'
                            .format(i + 1, xml_file))
                        valid_objs_num -= 1
                        continue
                    if obj.find('bndbox') is None:
                        print(
                            '[VOC check ] bndbox of {}-th object does not exist in {}'
                            .format(i + 1, xml_file))
                        valid_objs_num -= 1
                        continue
                    if obj.find('bndbox').find('xmin') is None:
                        print(
                            '[VOC check ] xmin of {}-th object does not exist in {}'
                            .format(i + 1, xml_file))
                        valid_objs_num -= 1
                        continue
                    x1 = float(obj.find('bndbox').find('xmin').text)
                    if obj.find('bndbox').find('ymin') is None:
                        print(
                            '[VOC check ] ymin of {}-th object does not exist in {}'
                            .format(i + 1, xml_file))
                        valid_objs_num -= 1
                        continue
                    y1 = float(obj.find('bndbox').find('ymin').text)
                    if obj.find('bndbox').find('xmax') is None:
                        print(
                            '[VOC check ] xmax of {}-th object does not exist in {}'
                            .format(i + 1, xml_file))
                        valid_objs_num -= 1
                        continue
                    x2 = float(obj.find('bndbox').find('xmax').text)
                    if obj.find('bndbox').find('ymax') is None:
                        print(
                            '[VOC check ] ymax of {}-th object does not exist in {}'
                            .format(i + 1, xml_file))
                        valid_objs_num -= 1
                        continue
                    y2 = float(obj.find('bndbox').find('ymax').text)
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(im_w - 1, x2)
                    y2 = min(im_h - 1, y2)
                    if x2 <= x1:
                        print('[VOC check ] xmax of {}-th object is set lower than or equal to xmin in {}'.format(\
                              i+1, xml_file))
                        valid_objs_num -= 1
                        continue
                    if y2 <= y1:
                        print('[VOC check ] ymax of {}-th object is set lower than or equal to ymin in {}'.format(\
                              i+1, xml_file))
                        valid_objs_num -= 1
                        continue
                if valid_objs_num > 0:
                    records += 1
        if records <= 0:
            print('not found any voc record in %s' % (anno_path))
            formatcorrect = False
    return formatcorrect


def voc_analysis(data_dir):
    if voc_primary_analysis(data_dir):
        if voc_intermediate_analysis(data_dir):
            return True
    return False


def data_analysis(dataset_type, input_dir):
    try:
        assert dataset_type in ['VOC', 'COCO']
    except AssertionError as e:
        print('Now only support the VOC dataset and COCO dataset!!')
        os._exit(0)
    try:
        assert os.path.exists(input_dir)
    except AssertionError as e:
        print('The dataset folder does not exist!')

    if dataset_type == 'VOC':
        return voc_analysis(input_dir)


if __name__ == '__main__':
    flag = data_analysis('VOC', '/ssd2/cts_data')
    print(flag)
