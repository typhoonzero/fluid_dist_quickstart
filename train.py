#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import print_function

import paddle.fluid.core as core
import math
import os
import sys
import unittest

import numpy

import paddle
import paddle.fluid as fluid

BATCH_SIZE = 64


def loss_net(hidden, label):
    prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
    loss = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_loss = fluid.layers.mean(loss)
    acc = fluid.layers.accuracy(input=prediction, label=label)
    return prediction, avg_loss, acc


def mlp(img, label):
    hidden = fluid.layers.fc(input=img, size=200, act='tanh')
    hidden = fluid.layers.fc(input=hidden, size=200, act='tanh')
    return loss_net(hidden, label)


def conv_net(img, label):
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=img,
        filter_size=5,
        num_filters=20,
        pool_size=2,
        pool_stride=2,
        act="relu")
    conv_pool_1 = fluid.layers.batch_norm(conv_pool_1)
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        pool_size=2,
        pool_stride=2,
        act="relu")
    return loss_net(conv_pool_2, label)


def train(nn_type, use_cuda):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return
    img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    if nn_type == 'mlp':
        net_conf = mlp
    else:
        net_conf = conv_net

    prediction, avg_loss, acc = net_conf(img, label)
    optimizer = fluid.optimizer.Adam(learning_rate=0.001)
    optimizer.minimize(avg_loss)

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.mnist.train(), buf_size=500),
        batch_size=BATCH_SIZE)
    test_reader = paddle.batch(
        paddle.dataset.mnist.test(), batch_size=BATCH_SIZE)
    feeder = fluid.DataFeeder(feed_list=[img, label], place=place)

    def train_loop(main_program):
        
        startup_exe = fluid.Executor(place)
        startup_exe.run(fluid.default_startup_program())

        st = fluid.ExecutionStrategy()
        st.num_threads = 1
        st.allow_op_delay = False
        exe = fluid.ParallelExecutor(use_cuda, avg_loss.name,
            exec_strategy=st)

        for pass_id in range(100):
            for batch_id, data in enumerate(train_reader()):
                loss, = exe.run([avg_loss.name], feed=feeder.feed(data))
                print(loss)

    
    # port = os.getenv("PADDLE_PSERVER_PORT", "6174")
    # pserver_ips = os.getenv("PADDLE_PSERVER_IPS")  # ip,ip...
    # eplist = []
    # for ip in pserver_ips.split(","):
    #     eplist.append(':'.join([ip, port]))
    # pserver_endpoints = ",".join(eplist)  # ip:port,ip:port...
    pserver_endpoints = os.getenv("PADDLE_PSERVER_ENDPOINTS")
    trainers = int(os.getenv("PADDLE_TRAINERS"))
    # current_endpoint = os.getenv("POD_IP") + ":" + port
    current_endpoint = os.getenv("PADDLE_CURRENT_ENDPOINT")
    trainer_id = int(os.getenv("PADDLE_TRAINER_ID"))
    training_role = os.getenv("PADDLE_TRAINING_ROLE", "TRAINER")
    t = fluid.DistributeTranspiler()
    t.transpile(trainer_id, pservers=pserver_endpoints, trainers=trainers)
    if training_role == "PSERVER":
        pserver_prog = t.get_pserver_program(current_endpoint)
        pserver_startup = t.get_startup_program(current_endpoint,
                                                pserver_prog)
        ps_exe = fluid.Executor(fluid.CPUPlace())
        ps_exe.run(pserver_startup)
        ps_exe.run(pserver_prog)
    elif training_role == "TRAINER":
        train_loop(t.get_trainer_program())



if __name__ == '__main__':
    train("conv", True)

