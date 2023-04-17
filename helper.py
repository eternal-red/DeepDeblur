from __future__ import print_function
#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Helper

"""

import tensorflow as tf
from keras.backend import set_session
from keras import backend as K
import os

def init_gpu(gpu_id="0"):

	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

	config = tf.compat.v1.ConfigProto()
	config.gpu_options.allow_growth=True
	set_session(tf.compat.v1.Session(config=config))