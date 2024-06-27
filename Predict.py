import math
import os
import librosa
import numpy as np
import tensorflow.keras as keras
import soundfile as sf

FILE_PATH = 'analisys.wav'
MODEL_PATH = 'data/trained_model'
SAMPLE_RATE = 22050
DURATION = 1
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
accuracy_threshold = 0.8


def preprocess(FILE_PATH,  num_segments, n_mfcc=13, n_fft=2048, hop_length=512):

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)

    file_path = os.path.join(FILE_PATH)
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

    print("mfcc shape: {}".format(mfcc.shape))

    mfcc = mfcc[..., np.newaxis]
    return mfcc


def read_class_labels(file_path='data\labels.txt'):

    class_labels = {}
    with open(file_path, 'r') as f:
        for line in f:
            index, label = line.strip().split(': ')
            class_labels[int(index)] = label
    return class_labels


def check_accuracy(predictions):
    max_acc = None
    max_index = None
    for i, (acc) in enumerate(predictions[0]):
        if acc > accuracy_threshold:
            max_acc = acc
            max_index = i
    return max_acc, max_index


def predict(model, file_mfcc):

    file_mfcc = file_mfcc[np.newaxis, ...]
    print("file_mfcc shape: {}".format(file_mfcc.shape))
    predictions = model.predict(file_mfcc)
    print(predictions)
    checked_accuracy, checked_index = check_accuracy(predictions)
    if checked_index is not None:
        print("Checked: {}, {}".format(checked_accuracy, checked_index))
    else:
        print("The Audio input doesn't belong to any category!")
    return checked_index


def predictres():
    results = []
    class_labels = read_class_labels()
    print("File name: {}".format(FILE_PATH))

    signal, sr = librosa.load(FILE_PATH, sr=SAMPLE_RATE)
    total_duration = librosa.get_duration(y=signal, sr=SAMPLE_RATE)
    print(total_duration)
    print(len(signal))

    model = keras.models.load_model(MODEL_PATH)

    if total_duration > 1:
        num_segments = math.floor(total_duration)
    else:
        num_segments = math.ceil(total_duration)

    print(num_segments)

    for s in range(num_segments):
        print("Iteration", s)
        start_sample = int(SAMPLE_RATE * s)
        finish_sample = int(start_sample + SAMPLE_RATE)
        print("start_sample", start_sample)
        print("finish_sample", finish_sample)
        segment = signal[start_sample:finish_sample]
        segment_file_path = os.path.join('segmented', f'segment_{s}.wav')
        sf.write(segment_file_path, segment, sr)
        mfcc_segment = preprocess(segment_file_path, 1)

        print(f"Predicting segment {s + 1}/{num_segments}")
        prediction = predict(model, mfcc_segment)
        if prediction is not None:
            class_name = class_labels[prediction]
        else:
            class_name = "unknown"
        results.append([f"Segment {s + 1}", class_name])
        os.remove(segment_file_path)

    return results
