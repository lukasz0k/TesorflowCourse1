import os
import ssl
import glob
import shutil
import logging
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

logger = tf.get_logger().setLevel(logging.ERROR)
ssl._create_default_https_context = ssl._create_unverified_context

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()

def plotTraining(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(EPOCHS)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Loss")
    plt.savefig('./foo.png')
    plt.show()

_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
WORKING_DIR = './flower_data'
BATCH_SIZE = 100
IMG_SHAPE = 150
EPOCHS = 500

if not os.path.exists(WORKING_DIR):
    zip_path = tf.keras.utils.get_file(origin=_URL, fname='flower_photos.tgz', extract=False)
    shutil.unpack_archive(zip_path, WORKING_DIR)

base_dir = os.path.join(WORKING_DIR, 'flower_photos')
classes = ['roses', 'daisy', 'dandelion', 'sunflowers', 'tulips']

for cl in classes:
    img_path = os.path.join(base_dir, cl)
    if not os.path.exists(img_path):  # This class has already been moved
        continue

    images = glob.glob(img_path + '/*.jpg')
    print("{}: {} Images".format(cl, len(images)))

    train, val = images[:round(len(images) * 0.8)], images[round(len(images) * 0.8):]

    train_class_dir = os.path.join(base_dir, 'train', cl)
    val_class_dir = os.path.join(base_dir, 'val', cl)

    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir, exist_ok=True)

    for t in train:
        shutil.move(t, train_class_dir)

    for v in val:
        shutil.move(v, val_class_dir)

train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

train_image_generator = ImageDataGenerator(rescale=1./255,
                                           rotation_range=40,
                                           width_shift_range=-.2,
                                           height_shift_range=0.2,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True,
                                           fill_mode='nearest')
train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_SHAPE, IMG_SHAPE),
                                                           class_mode='sparse')

augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

validation_image_generator = ImageDataGenerator(rescale=1./255)
val_data_gen = validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                           directory=val_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_SHAPE, IMG_SHAPE),
                                                           class_mode='sparse')

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMG_SHAPE, IMG_SHAPE, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(5)
])
model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
model.summary()

history = model.fit(
    train_data_gen,
    steps_per_epoch=int(np.ceil(len(train_data_gen)/BATCH_SIZE)),
    epochs=EPOCHS,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(len(val_data_gen)/BATCH_SIZE))
)

plotTraining(history)
print(train_data_gen.class_indices)
