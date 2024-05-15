
import numpy as np
import tensorflow as tf
import os
from PIL import Image
from pathlib import Path

class_dict = {
    '2': 0,
    '3': 1,
    '4': 2,
    '5': 3,
    '6': 4,
    '7': 5,
    '8': 6,
    '9': 7,
    '10': 8,
    '20': 9,
    '40': 10,
    '50': 11,
    '60': 12,
    '70': 13,
    '80': 14,
    '90': 15
}


def gen_series(path):
    try:
        path = path.decode('utf-8')  # Assuming 'utf-8' encoding
    except:
        pass
    path = Path(path)
    path_train = path / 'train'
    
    classes = list(path_train.iterdir())

    image_paths = []
    for clas in classes:
        files = list(clas.iterdir())
        image_paths += files
    
    for image_path in image_paths:
        # get the label 
        label_key = image_path._cparts[2]
        label = class_dict[label_key]

        image = Image.open(image_path)
        image = image.resize(size=(224,224))
        img = np.array(image)/255

        # you do the image transformations
        yield img, label

# x_train = gen_series('data')

# sample = next(x_train)

x_train = tf.data.Dataset.from_generator(
    gen_series,
    args=['data'],
    output_signature=(
            tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),  # Image tensor
            tf.TensorSpec(shape=(), dtype=tf.int32)  # Label tensor
        )   
)

# x_train = iter(x_train)

# for i in range(1000):

#     try:
#         img, lbl = next(x_train)
#     except:
#         continue

#     print(lbl)

