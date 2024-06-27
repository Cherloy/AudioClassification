import os
import numpy as np
import soundfile as sf
import random
from scipy.signal import resample


def load_wav_file(filepath):
    data, sr = sf.read(filepath)
    return data, sr


def save_wav_file(data, sr, filepath):
    sf.write(filepath, data, sr)


def add_noise(data, noise_factor=0.005):
    noise = np.random.randn(len(data))
    return data + noise_factor * noise


def change_pitch(data, sr, n_steps):
    factor = 2 ** (n_steps / 12)
    data_resampled = resample(data, int(len(data) / factor))
    if len(data_resampled) > len(data):
        return data_resampled[:len(data)]
    else:
        return np.pad(data_resampled, (0, len(data) - len(data_resampled)), 'constant')


def change_speed(data, speed_factor):
    data_resampled = resample(data, int(len(data) / speed_factor))
    if len(data_resampled) > len(data):
        return data_resampled[:len(data)]
    else:
        return np.pad(data_resampled, (0, len(data) - len(data_resampled)), 'constant')


def augment_and_save(data, sr, filename, output_dir):
    augmentations = {
        "noise": add_noise,
        "pitch": change_pitch,
        "speed": change_speed,
    }

    # Сохраняем оригинальный файл
    save_wav_file(data, sr, os.path.join(output_dir, filename))

    # Применяем каждую аугментацию и сохраняем результат
    for aug_name, aug_func in augmentations.items():
        if aug_name == "pitch":
            augmented_data = aug_func(data, sr, n_steps=random.randint(-2, 2))
        elif aug_name == "speed":
            augmented_data = aug_func(data, speed_factor=random.uniform(0.9, 1.1))
        else:
            augmented_data = aug_func(data)

        new_filename = f"{os.path.splitext(filename)[0]}_{aug_name}.wav"
        save_wav_file(augmented_data, sr, os.path.join(output_dir, new_filename))


def process_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename in os.listdir(input_dir):
        if filename.endswith(".wav"):
            print(f"Обрабатывается файл: {filename}")
            filepath = os.path.join(input_dir, filename)
            data, sr = load_wav_file(filepath)
            augment_and_save(data, sr, filename, output_dir)


# Путь к исходным и целевым папкам
input_dir_creator = 'dataset/creator'
input_dir_other = 'dataset/other'
output_dir_creator = 'dataset/creators'
output_dir_other = 'dataset/others'

# Аугментация данных
process_directory(input_dir_creator, output_dir_creator)
process_directory(input_dir_other, output_dir_other)
