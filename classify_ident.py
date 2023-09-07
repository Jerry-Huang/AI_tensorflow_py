import shutil
import sys

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QFont, QPixmap
from PyQt5.QtWidgets import QApplication, QTabWidget, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, \
    QFileDialog


class Window(QTabWidget):

    IMG_H, IMG_W = 224, 224

    def __init__(self):
        super().__init__()
        self.setWindowTitle('猫狗识别')
        # 模型初始化
        self.model = tf.keras.models.load_model("./cnn_fv.keras")
        self.class_names = ['猫', '狗']
        self.resize(900, 700)
        self.img_label = QLabel()
        self.result = QLabel("等待识别")
        self.result.setFont(QFont('楷体', 24))
        self.initUI()

    def initUI(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        font = QFont('楷体', 15)

        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.img_label, 1, Qt.AlignCenter)
        left_widget.setLayout(left_layout)
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        btn_change = QPushButton(" 上传图片 ")
        btn_change.clicked.connect(self.change_img)
        btn_change.setFont(font)
        btn_predict = QPushButton(" 开始识别 ")
        btn_predict.setFont(font)
        btn_predict.clicked.connect(self.predict_img)
        right_layout.addStretch()
        right_layout.addWidget(self.result, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addStretch()
        right_layout.addWidget(btn_change)
        right_layout.addWidget(btn_predict)
        right_layout.addStretch()
        right_widget.setLayout(right_layout)
        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget)
        main_widget.setLayout(main_layout)

        self.addTab(main_widget, '主页')

    def change_img(self):
        openfile_name = QFileDialog.getOpenFileName(self, 'chose files', '',
                                                    'Image files(*.jpg *.png *jpeg)')  # 打开文件选择框选择文件
        img_name = openfile_name[0]  # 获取图片名称
        if img_name == '':
            pass
        else:
            target_image_name = "./tmp_up." + img_name.split(".")[-1]  # 将图片移动到当前目录
            shutil.copy(img_name, target_image_name)
            img_init = cv2.imread(target_image_name)
            h, w, c = img_init.shape
            scale = 400 / h
            img_show = cv2.resize(img_init, (0, 0), fx=scale, fy=scale)  # 将图片的大小统一调整到400的高，方便界面显示
            cv2.imwrite("show.png", img_show)
            img_init = cv2.resize(img_init, (Window.IMG_W, Window.IMG_H))  # 将图片大小调整到224*224用于模型推理
            cv2.imwrite('target.png', img_init)
            self.img_label.setPixmap(QPixmap("show.png"))
            self.result.setText("等待识别")

    def predict_img(self):
        img = Image.open('target.png')  # 读取图片
        img = np.asarray(img)  # 将图片转化为numpy的数组
        outputs = self.model.predict(img.reshape(1, Window.IMG_W, Window.IMG_H, 3))
        result_index = int(np.argmax(outputs))
        print(result_index)

        result = self.class_names[result_index]
        self.result.setText(result)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    x = Window()
    x.show()
    sys.exit(app.exec_())
