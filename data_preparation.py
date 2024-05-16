
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers

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

batch_size = 32
total_data = 2668

steps_per_epoch = total_data//batch_size

def load_and_preprocess_image(image_path, problematic_images):
    try:
        with Image.open(image_path) as image:
            image = image.resize((299, 299)).convert('RGB')  # Resize to 299x299 and convert to RGB
            img = np.array(image) / 255.0
        return img
    except (OSError, IOError, ValueError) as e:
        print(f"Error loading or processing image '{image_path}': {e}")
        problematic_images.append(image_path)  
        return None

def gen_series(path, batch_size=2, shuffle = True):
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
    
    # shuffle the image paths list
    if shuffle:
        pass
    num_data = len(image_paths)
    print(f'the number of images is {num_data}')

    n_batches = num_data//batch_size
    
    for batch in range(n_batches):
        imgs = []
        lbls = []
        for i in range(batch*batch_size, batch*batch_size+batch_size):
            image_path = image_paths[i]
            # get the label 
            label_key = image_path._cparts[2]
            label = class_dict[label_key]

            image = Image.open(image_path)
            image = image.resize((299, 299)).convert('RGB')  # Resize to 299x299 and convert to RGB
            # image = image.convert('RGB')
            img = np.array(image)/255
            imgs.append(img)
            lbls.append(label)
        
        imgs = np.array(imgs)
        lbls = np.array(lbls)
        yield imgs, lbls

trrain_data = gen_series('data') 


model = keras.applications.InceptionV3(include_top=True)
# print(model.summary())

base_inputs = model.layers[0].output 
base_outputs = model.layers[-2].output
classifier = layers.Dense(3)(base_outputs)
new_model = keras.Model(inputs=base_inputs, outputs=classifier)
new_model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# print(new_model.summary())
new_model.fit(trrain_data, epochs=3, steps_per_epoch=steps_per_epoch)

# new_model.evaluate(trrain_data, epochs=3, verbose=2)


