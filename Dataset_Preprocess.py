import json
import os
import librosa
import math
import numpy as np

DATASET_PATH = "data\dataset"
JSON_PATH = "data\dataset_output.json"

SAMPLE_RATE = 22050
DURATION = 1
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION


class DatasetError(Exception):
    pass


def check_dataset(dataset_path):
    if not os.path.isdir(dataset_path):
        raise DatasetError("Выбранный путь не является директорией")

    subdirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    if len(subdirs) < 2:
        raise DatasetError("Датасет должен состоятть минимум из 2-х классов")

    for subdir in subdirs:
        subdir_path = os.path.join(dataset_path, subdir)
        if not all(f.endswith('.wav') for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))):
            raise DatasetError(f"Поддиректория '{subdir}' содержит файлы в формате отличном от wav")


def save_mfcc(dataset_path, json_path, num_segments, n_mfcc=13, n_fft=2048, hop_length=512):

    TOTAL_CATEGORY = 0
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        if dirpath is not dataset_path:
            dirpath_components = dirpath.split("\\")
            semantic_label = dirpath_components[-1]
            label_str = f"{TOTAL_CATEGORY}: {semantic_label}"
            data["mapping"].append(label_str)
            TOTAL_CATEGORY = TOTAL_CATEGORY + 1
            print("\n Processing {}".format(semantic_label))

            for f in filenames:

                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                if len(signal) < SAMPLES_PER_TRACK:

                    pad_length = SAMPLES_PER_TRACK - len(signal)
                    signal = np.pad(signal, (0, pad_length), 'constant')

                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s
                    finish_sample = start_sample + num_samples_per_segment

                    mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample],
                                                sr=sr, n_mfcc=n_mfcc, n_fft=n_fft,
                                                hop_length=hop_length)
                    mfcc = mfcc.T

                    if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i - 1)
                        print("{}, segment: {}".format(file_path, s+1))

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

    print("Total category: {}".format(TOTAL_CATEGORY))

    class_labels = {}
    for index, folder_name in enumerate(sorted(os.listdir(dataset_path))):
        folder_path = os.path.join(dataset_path, folder_name)
        if os.path.isdir(folder_path):
            class_labels[index] = folder_name

    with open('data\labels.txt', 'w') as f:
        for index, label in class_labels.items():
            f.write(f"{index}: {label}\n")


def mfccmake(dataset_path):
    save_mfcc(dataset_path, JSON_PATH, 1)
