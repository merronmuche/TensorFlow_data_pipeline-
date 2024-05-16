
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
        image = image.resize((299, 299)).convert('RGB')  # Resize to 299x299 and convert to RGB

        image = image.resize(size=(299,299))
        img = np.array(image)/255

        # you do the image transformations
        yield img, label

# y  = gen_series('data')

# for i in y:
#     print(i)


x_train = tf.data.Dataset.from_generator(
    gen_series,
    args=['data'],
    output_signature=(
            tf.TensorSpec(shape=(299, 299, 3), dtype=tf.float32),  # Image tensor
            tf.TensorSpec(shape=(), dtype=tf.int32)  # Label tensor
        )   
)

x_train = x_train.shuffle(20).padded_batch(32)

# ids, sequence_batch = next(iter(x_train))
# print(ids.numpy())
# print()
# print(sequence_batch.numpy())

# for imgs, lbls in x_train:
#         print(f"Label: {lbls}")


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


# x_test
def gen_series(path):
    try:
        path = path.decode('utf-8')  
    except:
        pass
    path = Path(path)
    path_train = path / 'test'
    
    classes = list(path_train.iterdir())
    class_dict = {str(clas).split('/')[-1]: i for i, clas in enumerate(classes)}

    image_paths = []
    for clas in classes:
        files = list(clas.iterdir())
        image_paths += files
    
    problematic_images = [] 
    
    for image_path in image_paths:
        # Get the label
        label_key = image_path.parts[-2]
        label = class_dict.get(label_key, -1)  

        # Load and preprocess the image
        img = load_and_preprocess_image(image_path, problematic_images)
        if img is not None:
            yield img, label

    return problematic_images 


# Create a TensorFlow dataset from the generator function
x_test = tf.data.Dataset.from_generator(
    gen_series,
    args=['data'],
    output_signature=(
        tf.TensorSpec(shape=(299, 299, 3), dtype=tf.float32),  # Image tensor
        tf.TensorSpec(shape=(), dtype=tf.int32)  # Label tensor
    )
)

# # Test the dataset by iterating through it
# for img, lbl in x_test.take(10):  
#     print(f"Label: {lbl}")

# # Retrieve the list of problematic image paths after processing all images
# problematic_images = next(iter(x_test))

# # Review the problematic images and decide which ones to delete
# for image_path in problematic_images:
#     print(f"Problematic image: {image_path}")




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
new_model.fit(x_train, epochs=3, verbose=2)

