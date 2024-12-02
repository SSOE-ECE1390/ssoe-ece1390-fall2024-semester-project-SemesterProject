import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget, QHBoxLayout
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
import os
import cv2
from bokeh_and_overlay import bokeh_bg
from img_segmentation import icon_segmentation
from img_segmentation import person_segmentation
from img_overlay import img_overlayv2
from bokeh_effect import bokeh
from extra_features import resize


class BokehGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bokeh Swap")
        self.setGeometry(100, 100, 800, 600)
        
        main_layout = QVBoxLayout()
        image_layout = QHBoxLayout()
        button_layout = QHBoxLayout()

        self.input_label = QLabel("Input Image Here")
        self.icon_label = QLabel("Icon Image Here")
        self.background_label = QLabel("Background Image Here")
        self.output_label = QLabel("Processed Image Here")

        for label in [self.input_label, self.icon_label, self.background_label, self.output_label]:
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setFixedSize(200, 200)
            label.setStyleSheet("border: 1px solid black;")

        image_layout.addWidget(self.input_label)
        image_layout.addWidget(self.icon_label)
        image_layout.addWidget(self.background_label)
        image_layout.addWidget(self.output_label)

        self.upload_input_btn = QPushButton("Upload Input Image")
        self.upload_icon_btn = QPushButton("Upload Icon Image")
        self.upload_background_btn = QPushButton("Upload Background Image")
        self.process_btn = QPushButton("Process Images")

        button_layout.addWidget(self.upload_input_btn)
        button_layout.addWidget(self.upload_icon_btn)
        button_layout.addWidget(self.upload_background_btn)
        button_layout.addWidget(self.process_btn)

        self.upload_input_btn.clicked.connect(self.upload_input_image)
        self.upload_icon_btn.clicked.connect(self.upload_icon_image)
        self.upload_background_btn.clicked.connect(self.upload_background_image)
        self.process_btn.clicked.connect(self.process_images)

        main_layout.addLayout(image_layout)
        main_layout.addLayout(button_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # image paths
        self.input_image_path = None
        self.icon_image_path = None
        self.background_image_path = None

    def upload_input_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Input Image", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            self.input_image_path = file_path
            self.input_label.setPixmap(QPixmap(file_path).scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio))

    def upload_icon_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Icon Image", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            self.icon_image_path = file_path
            self.icon_label.setPixmap(QPixmap(file_path).scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio))

    def upload_background_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Background Image", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            self.background_image_path = file_path
            self.background_label.setPixmap(QPixmap(file_path).scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio))

    def process_images(self):
        if not all([self.input_image_path, self.icon_image_path, self.background_image_path]):
            self.output_label.setText("Please upload all images.")
            return
        output_path = os.path.relpath("Output/Overlay/processed_image.jpeg")
        icon_mask = icon_segmentation.segment_iconv2(self.icon_image_path)
        bokeh_background = bokeh_bg(self.input_image_path, self.icon_image_path, self.background_image_path, bokeh_selector=0)
        result = img_overlayv2.img_overlay(bokeh_background, self.icon_image_path, output_path="processed_image")
        try:
            self.output_label.setPixmap(QPixmap(output_path).scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio))
        except:
            pass
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BokehGUI()
    window.show()
    sys.exit(app.exec())

