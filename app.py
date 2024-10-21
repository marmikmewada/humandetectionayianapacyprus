import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt

class MotionDetectorApp(QWidget):
    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # Set up the GUI
        self.setWindowTitle("Human Detection in Video")
        self.setGeometry(100, 100, 800, 600)
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        self.setLayout(layout)

        # Set up a timer to update the video feed
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # Resize frame for better processing
        frame = cv2.resize(frame, (600, 400))
        orig_frame = frame.copy()
        
        # Detect humans in the frame
        boxes, weights = self.hog.detectMultiScale(frame, winStride=(8, 8))

        for (x, y, w, h) in boxes:
            cv2.rectangle(orig_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            print("Human detected!")

        # Convert the frame to QImage
        rgb_image = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_image))

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    video_path = input("Enter the video filename (e.g., napa.mp4): ")  # Input video file name
    window = MotionDetectorApp(video_path)
    window.show()
    sys.exit(app.exec_())
