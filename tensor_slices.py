
import tensorflow as tf


train, test = tf.keras.datasets.fashion_mnist.load_data()

images, labels = train
images = images/255

dataset = tf.data.Dataset.from_tensor_slices((images, labels))
dataset
one_el = next(iter(dataset))


for el in dataset:
    image, lable = el

print(one_el)
