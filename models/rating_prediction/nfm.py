#!/usr/bin/env python
"""
Implementation of Neural Factorization Machine.
Reference: Dziugaite, Gintare Karolina, and Daniel M. Roy. "Neural network matrix factorization." arXiv preprint arXiv:1511.06443 (2015).
"""

import tensorflow as tf
import time
import numpy as np

from Evaluation.RatingMetrics import *

__author__ = "Shuai Zhang"
__copyright__ = "Copyright 2018, The DeepRec Project"

__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Shuai Zhang"
__email__ = "cheungdaven@gmail.com"
__status__ = "Development"
