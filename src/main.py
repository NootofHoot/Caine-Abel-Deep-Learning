import torch
import datetime
from datetime import datetime
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
from transformers import pipeline

try:
    %tensorflow_version 2.x
except Exception:
    pass

%load_ext tensorboard

print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
    "This notebook requires TensorFlow 2.0 or above."

with file_writer.as_default():
  # Don't forget to reshape.
  images = np.reshape(train_images[0:25], (-1, 28, 28, 1))
  tf.summary.image("25 training data examples", images, max_outputs=60, step=0)

%tensorboard --logdir logs/train_data

# Caine and Abel classifactions 
from abel import 



from caine import 