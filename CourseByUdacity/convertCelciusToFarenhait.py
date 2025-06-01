import tensorflow as tf
import numpy as np
import logging
import matplotlib.pyplot as plt

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

celsius_q = np.array([-40, -10, 0, 8, 15, 22, 38], dtype = float)
fahrenheit_a = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

for i,c in enumerate(celsius_q):
    print("{} degree Celsius = {} degree Fahrenhet".format(c, fahrenheit_a[i]))

l_input = tf.keras.Input(shape=(1,))
l0 = tf.keras.layers.Dense(units=1)

model = tf.keras.Sequential([l_input, l0])
model.compile(loss='mean_squared_error', optimizer = tf.keras.optimizers.Adam(0.1))
history = model.fit(celsius_q, fahrenheit_a, epochs = 500, verbose=False)
print("Finished training the model")

plt.xlabel('Epoch Number')
plt.ylabel('Loss Magnitude')
plt.plot(history.history['loss'])
plt.show()

print(model.predict(np.array([100.0])))

print("These are the layer 0 variables: {}".format(l0.get_weights()))
