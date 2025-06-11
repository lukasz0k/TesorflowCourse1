import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

tfds.disable_progress_bar()

from tensorflow.keras import layers

(train_examples, validation_examples), info = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:]'],
    with_info=True,
    as_supervised=True
)

def format_image(image, label):
    image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0
    return image, label

num_examples = info.splits['train'].num_examples

BATCH_SIZE = 32
IMAGE_RES = 224
EPOCHS = 5

train_batches = train_examples.cache().shuffle(num_examples//4).map(format_image).batch(BATCH_SIZE).prefetch(1)
validation_batches = validation_examples.cache().map(format_image).batch(BATCH_SIZE).prefetch(1)

URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
feature_extractor = tf.keras.layers.Lambda(lambda x: hub.KerasLayer(URL,
                                                                  input_shape=(IMAGE_RES, IMAGE_RES,3))(x),
                                         input_shape=(IMAGE_RES, IMAGE_RES,3))
feature_extractor.trainable = False

model = tf.keras.Sequential([
    feature_extractor,
    layers.Dense(2)
])
model.summary()

model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

history = model.fit(train_batches,
                   epochs = EPOCHS,
                    validation_data = validation_batches)

class_names = np.array(info.features['label'].names)
class_names

image_batch, label_batch = next(iter(train_batches.take(1)))
image_batch = image_batch.numpy()
label_batch = label_batch.numpy()

predicted_batch = model.predict(image_batch)
predicted_batch = tf.squeeze(predicted_batch).numpy()
predicted_ids = np.argmax(predicted_batch, axis=-1)
predicted_class_names = class_names[predicted_ids]
predicted_class_names

print("Labels: ", label_batch)
print("Predicted labels: ", predicted_ids)

plt.figure(figsize=(10,9))
for n in range(30):
    plt.subplot(6,5,n+1)
    plt.imshow(image_batch[n])
    color = "blue" if predicted_ids[n] == label_batch[n] else "red"
    plt.title(predicted_class_names[n].title(), color=color)
    plt.axis('off')
_ = plt.suptitle("Model predictions (blue: correct, red: incorrect")
plt.show()

t = time.time()

export_path_keras = "./{}.h5".format(int(t))
print(export_path_keras)
model.save(export_path_keras)

reloaded = tf.keras.models.load_model(
    export_path_keras,
    custom_objects={'KerasLayes': hub.KerasLayer}
)
reloaded.summary()

result_batch = model.predict(image_batch)
reloaded_result_batch = reloaded.predict(image_batch)
(abs(result_batch - reloaded_result_batch)).max()

history = reloaded.fit(train_batches,
                       epochs=EPOCHS,
                       validation_data=validation_batches)

t = time.time()
export_path_sm = "./{}".format(int(t))
print(export_path_sm)
tf.saved_model.save(model, export_path_sm)

reloaded_sm = rf.saved_model.load(export_path_sm)
reloaded_sm_result_batch = reloaded_sm(image_batch, training=False).numpy()
(abs(result_batch - reloaded_result_batch)).max()

t = time.time()
export_path_sm = "./{}".format(int(t))
print(export_path_sm)
tf.saved_model.save(model, export_path_sm)

reload_sm_keras = tf.keras.models.load_model(
export_path_sm,
custom_objects={'KerasLayer': hub.KerasLayer})
reload_sm_keras.summary()

result_batch = model.predict(image_batch)
reload_sm_keras_result_batch = reload_sm_keras.predict(image_batch)
(abs(result_batch - reload_sm_keras_result_batch)).max()

try:
  from google.colab import files
  files.download('./model.zip')
except ImportError:
  pass