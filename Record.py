import pyaudio
import wave
import numpy as np
from PyQt5.QtCore import pyqtSignal, QObject


class AudioRecorder(QObject):
    data_ready = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.frames = []
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False

    def start_recording(self):
        self.is_recording = True
        self.frames = []
        self.stream = self.p.open(format=self.format,
                                  channels=self.channels,
                                  rate=self.rate,
                                  input=True,
                                  frames_per_buffer=self.chunk)
        self.record()

    def record(self):
        while self.is_recording:
            try:
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                self.frames.append(data)
                audio_data = np.frombuffer(data, dtype=np.int16)
                self.data_ready.emit(audio_data)  # Отправляем сигнал с данными
            except IOError as e:
                print(f"Error recording: {e}")

    def stop_recording(self):
        self.is_recording = False
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        self.save_recording()

    def save_recording(self):
        audio_data = np.frombuffer(b''.join(self.frames), dtype=np.int16)
        filtered_data = self.apply_filter(audio_data)
        wf = wave.open('analisys.wav', 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(filtered_data.tobytes())
        wf.close()
        print("Recording saved as analisys.wav")

    def apply_filter(self, audio_data):
        from scipy.signal import butter, lfilter

        def butter_lowpass(cutoff, fs, order=5):
            nyquist = 0.5 * fs
            normal_cutoff = cutoff / nyquist
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            return b, a

        def lowpass_filter(data, cutoff, fs, order=5):
            b, a = butter_lowpass(cutoff, fs, order=order)
            y = lfilter(b, a, data)
            return y

        cutoff_frequency = 15000
        filtered_data = lowpass_filter(audio_data, cutoff_frequency, self.rate)
        return filtered_data.astype(np.int16)
