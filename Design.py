import sys
import numpy as np
import traceback
import os
from PyQt5 import uic
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QTableWidgetItem, QFileDialog, QMessageBox, QLabel
from PyQt5.QtGui import QMovie
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from Dataset_Preprocess import mfccmake, DatasetError, check_dataset
from Model import modelcreate
from Predict import predictres
from Record import AudioRecorder


class Worker(QThread):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    update_button = pyqtSignal(bool)
    update_label = pyqtSignal(str)

    def __init__(self, dataset_path):
        super().__init__()
        self.dataset_path = dataset_path

    def run(self):
        try:
            self.update_label.emit("Обработка датасета...\nПожалуйста, подождите")
            mfccmake(self.dataset_path)
            self.update_label.emit("")
            self.finished.emit()
            self.update_button.emit(True)
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            self.error.emit(error_message)


class Workermodel(QThread):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    update_label = pyqtSignal(str)
    update_button = pyqtSignal(bool)
    training_completed = pyqtSignal(object)
    training_page = pyqtSignal(int)

    def __init__(self, label, line_edit, line_edit_2):
        super().__init__()
        self.label = label
        self.line_edit = line_edit
        self.line_edit_2 = line_edit_2

    def run(self):
        try:
            epochs = int(self.line_edit.text())
            batches = int(self.line_edit_2.text())

            if epochs == 0 or epochs > 50:
                epochs = 50
            if batches == 0 or batches > 128:
                batches = 128

            self.update_label.emit("Обучение модели...\nПожалуйста, подождите")

            history = modelcreate(epochs, batches)
            self.update_label.emit("")

            self.training_page.emit(4)
            self.training_completed.emit(history)
            self.finished.emit()
            self.update_button.emit(True)
        except Exception as e:
            if 'invalid literal for int()' in str(e):
                error_message = f"Введите корректные данные"
                self.error.emit(error_message)
            else:
                error_message = f"An error occurred: {str(e)}"
                self.error.emit(error_message)


class Workerpredict(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def run(self):
        try:
            results = predictres()
            print(results)
            self.finished.emit(results)
        except Exception as e:
            error_message = f"An error occurred: {str(e)}\n{traceback.format_exc()}"
            self.error.emit(error_message)


class Workerrecord(QThread):
    data_ready = pyqtSignal(np.ndarray)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, audio_recorder):
        super().__init__()
        self.audio_recorder = audio_recorder

    def run(self):
        try:
            self.audio_recorder.data_ready.connect(self.data_ready.emit)
            self.audio_recorder.start_recording()
            self.finished.emit()
        except Exception as e:
            error_message = f"An error occurred: {str(e)}\n{traceback.format_exc()}"
            self.error.emit(error_message)

    def stop_recording(self):
        self.audio_recorder.stop_recording()


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi('data\Design.ui', self)

        #Первая страница, обработка датасета
        self.pushButton.clicked.connect(lambda: self.switch_page(0))
        self.pushButton_10.clicked.connect(self.select_dataset)
        self.pushButton_6.clicked.connect(self.process_dataset)
        self.pushButton_6.setEnabled(False)

        #Обучение модели
        self.pushButton_2.clicked.connect(lambda: self.switch_page(1))
        self.pushButton_7.clicked.connect(self.model)

        #Анализ аудио
        self.pushButton_3.clicked.connect(lambda: self.switch_page(2))
        self.pushButton_8.clicked.connect(self.predict)

        #Запись
        self.pushButton_9.clicked.connect(lambda: self.switch_page(3))
        self.pushButton_4.clicked.connect(self.recordstart)
        self.stop_button = self.pushButton_5
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_recording)

        #Параметры модели
        self.pushButton_11.clicked.connect(self.load_file_content)

        self.start_check_button_states()
        self.all_buttons = [child for child in self.findChildren(QPushButton)]
        self.button_states = {}

        self.loading_label = self.findChild(QLabel, "loadingLabel")
        self.loading_label_2 = self.findChild(QLabel, "loadingLabel_2")

        self.worker = None
        self.audio_recorder = None
        self.worker_record = None

        #Вывод амплитуды
        plot_widget = self.findChild(QWidget, 'plotWidget')
        self.figure = Figure(facecolor='none')
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        layout = QVBoxLayout(plot_widget)
        layout.addWidget(self.canvas)
        self.seconds_to_display = 5
        self.sample_rate = 22050
        self.plot_data = np.zeros(self.seconds_to_display * self.sample_rate)
        self.line, = self.ax.plot(np.linspace(0, self.seconds_to_display, len(self.plot_data)), self.plot_data)
        self.ax.set_ylim([-32768, 32767])
        self.ax.axis('off')
        self.ax.set_facecolor('white')

        #Создание графиков для отрисовки точности
        self.figure_accuracy = plt.figure(figsize=(10, 5))
        self.ax_accuracy = self.figure_accuracy.add_subplot(111)
        self.ax_accuracy.set_title('Accuracy')
        self.ax_accuracy.set_xlabel('Epoch')
        self.ax_accuracy.set_ylabel('Accuracy')

        self.figure_loss = plt.figure(figsize=(10, 5))
        self.ax_loss = self.figure_loss.add_subplot(111)
        self.ax_loss.set_title('Loss')
        self.ax_loss.set_xlabel('Epoch')
        self.ax_loss.set_ylabel('Loss')

        self.canvas_accuracy = FigureCanvas(self.figure_accuracy)
        self.canvas_loss = FigureCanvas(self.figure_loss)

        self.canvas_accuracy.setFixedSize(400, 380)
        self.canvas_loss.setFixedSize(400, 380)

        plot_widget_accuracy = self.findChild(QWidget, 'plotWidgetAccuracy')
        plot_widget_loss = self.findChild(QWidget, 'plotWidgetLoss')

        self.canvas_accuracy.setParent(plot_widget_accuracy)
        self.canvas_loss.setParent(plot_widget_loss)

    def load_file_content(self):
        try:
            self.switch_page(5)
            # Открываем файл и считываем его содержимое
            with open("data\summary.txt", "r", encoding="utf-8") as file:
                content = file.read()

            # Отображаем содержимое файла в QTextEdit
            self.textEdit.setPlainText(content)
        except FileNotFoundError:
            self.textEdit.setPlainText("Файл не найден")

    def start_check_button_states(self):
        root_path = os.path.dirname(os.path.abspath(__file__))
        #Проверка присутствия обработанных данных
        dataset_path = os.path.join(root_path, "data\dataset_output.json")
        if os.path.exists(dataset_path):
            self.pushButton_2.setEnabled(True)
        else:
            self.pushButton_2.setEnabled(False)

        #Проверка сводки модели
        summary_path = os.path.join(root_path, "data\summary.txt")
        if os.path.exists(summary_path):
            self.pushButton_11.setEnabled(True)
        else:
            self.pushButton_11.setEnabled(False)

        #Проверка присутствия модели
        model_path = os.path.join(root_path, "data/trained_model", "saved_model.pb")
        if os.path.exists(model_path):
            self.pushButton_3.setEnabled(True)
        else:
            self.pushButton_3.setEnabled(False)

    def record_button_states(self):
        for button in self.all_buttons:
            self.button_states[button] = button.isEnabled()

    def select_dataset(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Dataset Directory")
        if directory:
            try:
                check_dataset(directory)
                self.label_2.setText(directory)
                self.pushButton_6.setEnabled(True)
            except DatasetError as e:
                self.label_2.setText("Путь не выбран корректно")
                self.show_message(str(e))

    def process_dataset(self):
        dataset_path = self.label_2.text()
        if dataset_path:
            movie = QMovie("data\loading.gif")
            self.loading_label_2.setMovie(movie)
            movie.start()
            self.record_button_states()
            self.set_buttons_enabled(False)
            self.worker = Worker(dataset_path)
            self.worker.update_label.connect(self.update_label_text)
            self.worker.update_button.connect(self.pushButton_2.setEnabled)
            self.worker.finished.connect(self.task_complete)
            self.worker.error.connect(self.show_message)
            self.worker.start()
        else:
            self.show_message("No dataset path selected.")

    def model(self):
        root_path = os.path.dirname(os.path.abspath(__file__))

        # Проверяем наличие файла dataset_output.json
        dataset_path = os.path.join(root_path, "data\dataset_output.json")
        if not os.path.exists(dataset_path):
            error_message = "Файл 'dataset_output.json' не найден в папке, пожалуйста сначала обработайте датасет."
            self.show_message(error_message)
            return

        movie = QMovie("data\loading.gif")
        self.loading_label.setMovie(movie)
        movie.start()

        self.record_button_states()
        self.set_buttons_enabled(False)
        self.worker = Workermodel(self.label, self.lineEdit, self.lineEdit_2)
        self.worker.update_button.connect(self.pushButton_3.setEnabled)
        self.worker.update_label.connect(self.update_label_text)
        self.worker.finished.connect(self.task_complete)
        self.worker.training_completed.connect(self.show_training_results)  # Connect to training_completed signal
        self.worker.training_page.connect(self.switch_page)
        self.worker.error.connect(self.show_message)
        self.worker.start()

    def predict(self):
        root_path = os.path.dirname(os.path.abspath(__file__))

        # Проверяем наличие файла analisys.wav
        audio_file_path = os.path.join(root_path, "analisys.wav")
        if not os.path.exists(audio_file_path):
            error_message = 'Файл "analisys.wav" не найден, пожалуйста пройдите в окно "Начать запись" для его создания.'
            self.show_message(error_message)
            return
        self.record_button_states()
        self.set_buttons_enabled(False)
        self.worker = Workerpredict()
        self.worker.finished.connect(self.update_table_widget)
        self.worker.start()

    def update_table_widget(self, results):
        self.tableWidget.setRowCount(0)
        for row_number, (segment, prediction) in enumerate(results):
            self.tableWidget.insertRow(row_number)
            self.tableWidget.setItem(row_number, 0, QTableWidgetItem(segment))
            self.tableWidget.setItem(row_number, 1, QTableWidgetItem(str(prediction)))
        self.task_complete()

    def recordstart(self):
        self.plot_data = np.zeros(self.seconds_to_display * self.sample_rate)
        self.line.set_ydata(self.plot_data)
        self.canvas.draw()
        self.audio_recorder = AudioRecorder()
        self.worker_record = Workerrecord(self.audio_recorder)
        self.record_button_states()
        self.set_buttons_enabled(False)
        self.stop_button.setEnabled(True)
        self.worker_record.data_ready.connect(self.update_plot)
        self.worker_record.start()
        self.worker_record.finished.connect(self.task_complete)

    def stop_recording(self):
        if self.worker_record:
            self.worker_record.stop_recording()
        self.stop_button.setEnabled(False)

    def task_complete(self):
        try:
            self.loading_label_2.clear()
            self.loading_label.clear()
        except:
            pass

        for button, state in self.button_states.items():
            button.setEnabled(state)  # Восстанавливаем исходное состояние кнопок
        self.stop_button.setEnabled(False)

    def set_buttons_enabled(self, enabled):
        self.pushButton.setEnabled(enabled)
        for button in self.all_buttons:
            button.setEnabled(enabled)
        self.stop_button.setEnabled(False)

    def show_message(self, message):
        msg_box = QMessageBox()
        msg_box.setWindowTitle("Error")
        msg_box.setText(message)
        msg_box.exec_()
        self.task_complete()

    @pyqtSlot(np.ndarray)
    def update_plot(self, data):
        scaled_data = data * 4
        num_new_points = len(scaled_data)
        self.plot_data = np.roll(self.plot_data, -num_new_points)
        self.plot_data[-num_new_points:] = scaled_data
        self.line.set_ydata(self.plot_data)
        self.canvas.draw()

    def switch_page(self, page_index):
        self.stackedWidget.setCurrentIndex(page_index)

    @pyqtSlot(object)
    def show_training_results(self, history):
        try:
            if history:
                # Извлеките значения accuracy и loss из history
                accuracy = history.history['accuracy']
                val_accuracy = history.history['val_accuracy']
                loss = history.history['loss']
                val_loss = history.history['val_loss']

                self.ax_accuracy.clear()
                self.ax_loss.clear()

                # Постройте графики на соответствующих подзаголовках
                self.ax_accuracy.plot(accuracy, label='Train Accuracy')
                self.ax_accuracy.plot(val_accuracy, label='Validation Accuracy')
                self.ax_accuracy.legend()
                self.canvas_accuracy.draw()

                self.ax_loss.plot(loss, label='Train Loss')
                self.ax_loss.plot(val_loss, label='Validation Loss')
                self.ax_loss.legend()
                self.canvas_loss.draw()

            else:
                self.show_message("Failed to retrieve training history.")
        except Exception as e:
            error_message = f"An error occurred: {str(e)}\n{traceback.format_exc()}"
            self.show_message(error_message)

    def update_label_text(self, text):
        self.label.setText(text)
        self.label_10.setText(text)

    @pyqtSlot(object, bool)
    def update_button(self, button, state):
        button.setEnabled(state)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())
