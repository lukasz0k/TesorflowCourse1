import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

keras = tf.keras

def plot_series(time, series, format="-", start=0, end=None, label=None):
    plt.plot(time[start:end], series[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    if label:
        plt.legend(fontsize=14)
    plt.grid(True)

def trend(time, slope=0):
    return slope * time

def seasonal_pattern(season_time):
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

def white_noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

def moving_average_forecast(series, window_size):
    mov = np.cumsum(series)
    mov[window_size:] = mov[window_size:] - mov[:-window_size]
    return mov[window_size - 1:-1] / window_size

time = np.arange(4 * 365 + 1)
slope = 0.05
baseline = 10
amplitude = 40
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)

noise_level = 5
noise = white_noise(time, noise_level, seed=42)
series += noise

plt.figure(figsize=(10,6))
plot_series(time, series)
plt.show()

split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

native_forcast = series[split_time - 1:-1]
plt.figure(figsize=(10,6))
plot_series(time_valid, x_valid, start=0, end=150, label="Series")
plot_series(time_valid, native_forcast, start=1, end=151, label="Forecast")
plt.show()

mae = keras.metrics.MeanAbsoluteError()
mae.update_state(x_valid, native_forcast)
print(mae.result().numpy())

moving_avg = moving_average_forecast(series, 30)[split_time - 30:]
plt.figure(figsize=(10,6))
plot_series(time_valid, x_valid, label="Series")
plot_series(time_valid, moving_avg, label="Moving average (30 days)")
plt.show()

mae = keras.metrics.MeanAbsoluteError()
mae.update_state(x_valid, native_forcast)
print(mae.result().numpy())

diff_series = (series[365:] - series[:-365])
diff_time = time[365:]
plt.figure(figsize=(10,6))
plot_series(diff_time, diff_series, label="Series(t) - Series(t-365)")
plt.show()

plt.figure(figsize=(10, 6))
plot_series(time_valid, diff_series[split_time - 365:], label="Series(t) – Series(t–365)")
plt.show()

diff_moving_avg = moving_average_forecast(diff_series, 50)[split_time - 365 - 50:]
plt.figure(figsize=(10,6))
plot_series(time_valid, diff_moving_avg, label="Moving Average of Diff")
plot_series(time_valid, diff_series[split_time - 365:], label="Series(t) - Series(t-365")
plt.show()

diff_moving_plus_past = series[split_time - 365:-365] + diff_moving_avg
plt.figure(figsize=(10,6))
plot_series(time_valid, x_valid, label="Series")
plot_series(time_valid, diff_moving_plus_past, label="Forecasts")
plt.show()

mae = keras.metrics.MeanAbsoluteError()
mae.update_state(x_valid, native_forcast)
print(mae.result().numpy())

diff_moving_plus_smooth_past = (moving_average_forecast(series[split_time - 370:-359], 11)
                                + diff_moving_plus_past)
plt.figure(figsize=(10,6))
plot_series(time_valid, x_valid, label="Series")
plot_series(time_valid, diff_moving_plus_smooth_past, label="Forecasts")
plt.show()

mae = keras.metrics.MeanAbsoluteError()
mae.update_state(x_valid, native_forcast)
print(mae.result().numpy())