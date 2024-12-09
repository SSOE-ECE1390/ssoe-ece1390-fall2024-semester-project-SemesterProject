from PyQt5.QtWidgets import QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QFileDialog, QComboBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import cv2 as cv
import numpy as np
from ApplyHairstyles import *

class ViewScreen(QWidget):
    def __init__(self, parent):
        '''
        This class displays the image and provides controls to apply hairstyles and filters.
        '''
        super().__init__()
        self.parent = parent
        self.current_image_path = None
        self.layout = QVBoxLayout()
        
        self.image_label = QLabel("No Image")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(400, 400)
        self.image_label.setStyleSheet("border: 1px solid black; background-color: white;")
        
        self.controls_layout = QVBoxLayout()
        self.combo_box = QComboBox()
        self.combo_box.addItems(["Select Hairstyle", "Bald", "Curly", "Mullet", "Dreadlocks", "Waves"])
        self.apply_btn = QPushButton("Apply Style")
        self.apply_btn.setStyleSheet("background-color: darkblue; color: white;")
        self.apply_btn.clicked.connect(self.apply_hairstyle)
        self.new_photo_btn = QPushButton("Capture New Photo")
        self.new_photo_btn.setStyleSheet("background-color: darkgreen; color: white;")
        self.new_photo_btn.clicked.connect(self.take_new_photo)
        self.upload_new_photo_btn = QPushButton("Upload New Photo")
        self.upload_new_photo_btn.setStyleSheet("background-color: darkorange; color: white;")
        self.upload_new_photo_btn.clicked.connect(self.upload_new_photo)
        
        # Filter buttons
        self.negative_btn = QPushButton("Negative")
        self.negative_btn.setStyleSheet("background-color: darkred; color: white;")
        self.negative_btn.clicked.connect(self.apply_negative_filter)
        
        choose_hairstyle_label = QLabel("Choose Hairstyle:")
        choose_hairstyle_label.setStyleSheet("font-size: 16px; color: lightblue;")
        
        self.controls_layout.addWidget(choose_hairstyle_label)
        self.controls_layout.addWidget(self.combo_box)
        self.controls_layout.addWidget(self.apply_btn)
        self.controls_layout.addWidget(self.new_photo_btn)
        self.controls_layout.addWidget(self.upload_new_photo_btn)
        
        # Add filter buttons to layout
        self.controls_layout.addWidget(self.negative_btn)
        
        self.layout.addWidget(self.image_label, alignment=Qt.AlignCenter)
        self.layout.addLayout(self.controls_layout)
        self.layout.setAlignment(Qt.AlignCenter)
        self.setLayout(self.layout)

    def set_image(self, image_path):
        '''
        Set the image to be displayed on the screen.

        :param image_path: The path to the image file.
        '''
        self.current_image_path = image_path
        pixmap = QPixmap(image_path).scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)

    def apply_hairstyle(self):
        '''
        Apply the selected hairstyle to the image.
        '''
        hairstyle = self.combo_box.currentText()
        input_image_path = "boxed_photo.jpg"
        output_image_path = "filtered_photo.jpg"
        if hairstyle == "Select Hairstyle":
            pass
        elif hairstyle == "Bald":
            apply_bald_hairstyle(input_image_path, output_image_path)
        elif hairstyle == "Curly":
            apply_curly_hairstyle(input_image_path, output_image_path)
        elif hairstyle == "Mullet":
            apply_mullet_hairstyle(input_image_path, output_image_path)
        elif hairstyle == "Dreadlocks":
            apply_dread_hairstyle(input_image_path, output_image_path)
        elif hairstyle == "Waves":
            apply_waves_hairstyle(input_image_path, output_image_path)

        if output_image_path:
            self.set_image(output_image_path)

    def take_new_photo(self):
        '''
        Switch to the camera screen to capture a new photo.
        '''
        self.parent.camera_screen.start_camera()
        self.parent.stacked_widget.setCurrentWidget(self.parent.camera_screen)

    def upload_new_photo(self):
        '''
        Upload a new photo from the file system.
        '''
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Photo", "", "Images (*.png *.jpg *.jpeg)", options=options
        )
        if file_path:
            #go back to bounding box edit
            self.parent.BoundingBox_edit_screen.inital_set_image(file_path,box=None)
            self.parent.stacked_widget.setCurrentWidget(self.parent.BoundingBox_edit_screen)

    def apply_negative_filter(self):
        '''
        Apply negative filter to the image.
        '''
        if self.current_image_path:
            image = cv.imread(self.current_image_path)
            negative_image = cv.bitwise_not(image)
            output_image_path = "negative_image.jpg"
            cv.imwrite(output_image_path, negative_image)
            self.set_image(output_image_path)
