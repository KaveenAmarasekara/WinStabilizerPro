import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel,
    QVBoxLayout, QFileDialog, QComboBox, QProgressBar, QMessageBox
)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal


class StabilizationThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)

    def __init__(self, input_path, strength):
        super().__init__()
        self.input_path = input_path
        self.strength = strength

    def run(self):
        try:
            cap = cv2.VideoCapture(self.input_path)
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            _, prev = cap.read()
            prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
            transforms = []

            for i in range(n_frames - 1):
                success, curr = cap.read()
                if not success:
                    break
                curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
                prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)
                curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

                valid = status.flatten() == 1
                prev_pts = prev_pts[valid]
                curr_pts = curr_pts[valid]

                m = cv2.estimateAffinePartial2D(prev_pts, curr_pts)[0]
                if m is None:
                    m = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

                dx = m[0, 2]
                dy = m[1, 2]
                da = np.arctan2(m[1, 0], m[0, 0])
                transforms.append([dx, dy, da])
                prev_gray = curr_gray

                self.progress.emit(int((i / n_frames) * 50))  # First 50%

            trajectory = np.cumsum(transforms, axis=0)
            smoothed_trajectory = self.smooth(trajectory, self.strength)
            difference = smoothed_trajectory - trajectory
            transforms_smooth = np.array(transforms) + difference

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            output_initialized = False
            out = None

            for i in range(n_frames - 1):
                success, frame = cap.read()
                if not success:
                    break

                dx, dy, da = transforms_smooth[i]
                m = np.array([
                    [np.cos(da), -np.sin(da), dx],
                    [np.sin(da),  np.cos(da), dy]
                ])

                frame_stabilized = cv2.warpAffine(frame, m, (w, h))
                frame_stabilized = self.crop_border(frame_stabilized)

                if frame_stabilized is None:
                    continue

                if not output_initialized:
                    h_out, w_out = frame_stabilized.shape[:2]
                    out_path = os.path.join(os.path.dirname(self.input_path), "stabilized_output.mp4")
                    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w_out, h_out))
                    output_initialized = True

                out.write(frame_stabilized)
                self.progress.emit(50 + int((i / n_frames) * 50))  # Next 50%

            cap.release()
            if output_initialized:
                out.release()
                self.finished.emit(out_path)
            else:
                self.finished.emit("Error: Output file not created.")

        except Exception as e:
            self.finished.emit(f"Error: {str(e)}")

    def smooth(self, trajectory, strength):
        radius_map = {"Low": 5, "Medium": 15, "High": 30}
        radius = radius_map.get(strength, 15)
        smoothed = np.copy(trajectory)
        for i in range(3):
            smoothed[:, i] = self.moving_average(trajectory[:, i], radius)
        return smoothed

    def moving_average(self, curve, radius):
        window_size = 2 * radius + 1
        filter = np.ones(window_size) / window_size
        curve_pad = np.pad(curve, (radius, radius), mode='edge')
        return np.convolve(curve_pad, filter, mode='same')[radius:-radius]

    def crop_border(self, frame, crop_percent=0.05):
        h, w = frame.shape[:2]
        crop_h, crop_w = int(h * crop_percent), int(w * crop_percent)
        return frame[crop_h:h - crop_h, crop_w:w - crop_w]


class VideoStabilizerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WinStabilizerPro - Video Stabilizer")
        self.setWindowIcon(QIcon('icon.png'))
        self.setFixedSize(400, 300)

        layout = QVBoxLayout()

        self.label = QLabel("Select a video file to stabilize:")
        layout.addWidget(self.label)

        self.select_button = QPushButton("Select Video")
        self.select_button.clicked.connect(self.select_file)
        layout.addWidget(self.select_button)

        self.strength_label = QLabel("Select Stabilization Strength:")
        layout.addWidget(self.strength_label)

        self.strength_combo = QComboBox()
        self.strength_combo.addItems(["Low", "Medium", "High"])
        layout.addWidget(self.strength_combo)

        self.start_button = QPushButton("Start Stabilization")
        self.start_button.clicked.connect(self.start_stabilization)
        layout.addWidget(self.start_button)

        self.progress = QProgressBar()
        layout.addWidget(self.progress)

        self.status = QLabel("")
        layout.addWidget(self.status)

        self.setLayout(layout)
        self.input_path = None

    def select_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov)")
        if path:
            self.input_path = path
            self.label.setText(f"Selected: {os.path.basename(path)}")

    def start_stabilization(self):
        if not self.input_path:
            QMessageBox.warning(self, "No File", "Please select a video file first.")
            return

        strength = self.strength_combo.currentText()
        self.thread = StabilizationThread(self.input_path, strength)
        self.thread.progress.connect(self.progress.setValue)
        self.thread.finished.connect(self.stabilization_done)
        self.progress.setValue(0)
        self.status.setText("Processing...")
        self.thread.start()

    def stabilization_done(self, message):
        if message.endswith(".mp4"):
            self.status.setText("Done! Output saved.")
            QMessageBox.information(self, "Success", f"Stabilized video saved as:\n{message}")
        else:
            self.status.setText("Failed.")
            QMessageBox.critical(self, "Error", message)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('icon.png'))
    window = VideoStabilizerApp()
    window.show()
    sys.exit(app.exec_())
