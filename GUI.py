import sys
import numpy as np


from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QStackedWidget, QGridLayout
from PyQt5.QtGui import QImage, QPixmap, QIntValidator
from PyQt5 import uic, QtCore

import ImageProcessor

# load interfaces
main_page_ui = 'UI/MainWindow.ui'
main_page_form, main_page_base = uic.loadUiType(main_page_ui)

cal_red_color_ui = 'UI/RedColorCalibrationWindow.ui'
cal_red_color_form, cal_red_color_base = uic.loadUiType(cal_red_color_ui)

cal_green_color_ui = 'UI/GreenColorCalibrationWindow.ui'
cal_green_color_form, cal_green_color_base = uic.loadUiType(cal_green_color_ui)

cal_yellow_color_ui = 'UI/YellowColorCalibrationWindow.ui'
cal_yelow_color_form, cal_yellow_color_base = uic.loadUiType(cal_yellow_color_ui)

cal_segmentation_ui = 'UI/SegmentationCalibrationWindow.ui'
cal_segmentation_form, cal_segmentation_base = uic.loadUiType(cal_segmentation_ui)

cal_snakes_ui = 'UI/SnakesCalibrationWindow.ui'
cal_snakes_form, cal_snakes_base = uic.loadUiType(cal_snakes_ui)

cal_threshold_ui = 'UI/ThresholdCalibrationWindow.ui'
cal_threshold_form, cal_threshold_base = uic.loadUiType(cal_threshold_ui)

output_protocol_ui = 'UI/OutputProtocolWindow.ui'
output_protocol_form, output_protocol_base = uic.loadUiType(output_protocol_ui)

parameters_ui = 'UI/ParametersWindow.ui'
parameters_form, parameters_base = uic.loadUiType(parameters_ui)

image_processor = ImageProcessor.ImageProcessor()


def format_image(image):
    qformat = QImage.Format_Indexed8
    if len(image.shape) == 3:  # rows, cols, channels
        if image.shape[2] == 4:  # 4 channels
            qformat = QImage.Format_RGBA8888
        else:
            qformat = QImage.Format_RGB888

    output = QImage(image, image.shape[1], image.shape[0], image.strides[0], qformat)
    return output.rgbSwapped()


class MainWindow(main_page_base, main_page_form):
    def __init__(self):
        super(main_page_base, self).__init__()
        self.setupUi(self)

        self.load_btn.clicked.connect(self.load_source)

        # disable buttons while image is loaded
        self.show_param_btn.setDisabled(True)
        self.analyze_btn.setDisabled(True)
        self.calibrate_btn.setDisabled(True)
        self.save_btn.setDisabled(True)
        self.refresh_btn.setDisabled(True)

        self.action_save.setDisabled(True)
        self.action_red.setDisabled(True)
        self.action_green.setDisabled(True)
        self.action_yellow.setDisabled(True)
        self.action_edge.setDisabled(True)
        self.action_active_contour.setDisabled(True)
        self.action_threshold.setDisabled(True)

        self.show_param_btn.clicked.connect(self.show_parameters)
        self.analyze_btn.clicked.connect(self.analyze)
        self.calibrate_btn.clicked.connect(self.start_calibration_process)
        self.format_btn.clicked.connect(self.format_output)
        self.save_btn.clicked.connect(self.save)
        self.refresh_btn.clicked.connect(self.refresh)

        self.action_load.triggered.connect(self.load_source)
        self.action_save.triggered.connect(self.save)
        self.action_exit.triggered.connect(self.exit_app)
        self.action_markup.triggered.connect(self.markup_style)
        self.action_output_format.triggered.connect(self.format_output)
        self.action_red.triggered.connect(self.red_calibration)
        self.action_green.triggered.connect(self.green_calibration)
        self.action_yellow.triggered.connect(self.yellow_calibration)
        self.action_edge.triggered.connect(self.segmentation_calibration)
        self.action_active_contour.triggered.connect(self.snake_calibration)
        self.action_threshold.triggered.connect(self.threshold_calibration)

    def load_source(self):
        self.file_names, filter = QFileDialog.getOpenFileNames(self, 'Open files', __file__, "Image Files (*.tif)")
        if self.file_names:
            self.show_param_btn.setDisabled(False)
            self.analyze_btn.setDisabled(False)
            self.calibrate_btn.setDisabled(False)
            self.format_btn.setDisabled(False)
            self.save_btn.setDisabled(False)
            self.action_save.setDisabled(False)
            self.action_red.setDisabled(False)
            self.action_green.setDisabled(False)
            self.action_yellow.setDisabled(False)
            self.action_edge.setDisabled(False)
            self.action_active_contour.setDisabled(False)
            self.action_threshold.setDisabled(False)
            self.refresh_btn.setDisabled(False)

            image = format_image(image_processor.load_origin(self.file_names[0]))
            self.input_picture_label.setPixmap(QPixmap.fromImage(image))
            self.input_picture_label.setScaledContents(True)
            self.input_picture_label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

            image = format_image(image_processor.full_output)
            self.output_picture_label.setPixmap(QPixmap.fromImage(image))
            self.output_picture_label.setScaledContents(True)
            self.output_picture_label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

    def refresh(self):
        image = format_image(image_processor.full_output)
        self.output_picture_label.setPixmap(QPixmap.fromImage(image))
        self.output_picture_label.setScaledContents(True)
        self.output_picture_label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

    def save(self):
        # TODO ulozi aktualny obrazok a protokol
        return

    def exit_app(self):
        sys.exit(1)

    def threshold_calibration(self):
        self.main = ThresholdCalibrationWindow()
        self.main.show()

    def snake_calibration(self):
        self.main = SnakesCalibrationWindow()
        self.main.show()

    def segmentation_calibration(self):
        self.main = SegmentationCalibrationWindow()
        self.main.show()

    def yellow_calibration(self):
        self.main = YellowColorCalibrationWindow()
        self.main.show()

    def green_calibration(self):
        self.main = GreenColorCalibrationWindow()
        self.main.show()

    def red_calibration(self):
        self.main = RedColorCalibrationWindow()
        self.main.show()

    def markup_style(self):
        # TODO formular pre volenie zobrazovania buniek
        return

    def show_parameters(self):
        self.main = ParameterWindow()
        self.main.show()

    # TODO analyzovat vsetko + progres form
    def analyze(self):
        for file in self.file_names:
            image_processor.load_origin(file)


    def start_calibration_process(self):
        self.main = CalibrationWindow()
        self.main.show()

    def format_output(self):
        self.main = OutputProtocolEditWindow()
        self.main.show()


class CalibrationWindow(QWidget):
    def __init__(self):
        super(CalibrationWindow, self).__init__()
        self.layout = QGridLayout()
        self.setMinimumSize(1000, 560)
        self.setWindowTitle('Kalibrácia')
        self.i = 0

        # Creating widgets
        self.redcal = RedColorCalibrationWindow()
        self.greencal = GreenColorCalibrationWindow()
        self.yellowcal = YellowColorCalibrationWindow()
        self.segmentcal = SegmentationCalibrationWindow()
        self.snakescal = SnakesCalibrationWindow()
        self.threshcal = ThresholdCalibrationWindow()

        # Stacking stack
        self.stack = QStackedWidget(self)
        self.stack.addWidget(self.redcal)
        self.stack.addWidget(self.greencal)
        self.stack.addWidget(self.yellowcal)
        self.stack.addWidget(self.segmentcal)
        self.stack.addWidget(self.snakescal)
        self.stack.addWidget(self.threshcal)
        self.layout.addWidget(self.stack)
        self.stack.setCurrentIndex(self.i)

        # Continue buttons actions
        self.redcal.continue_btn.clicked.connect(self.action_continue)
        self.greencal.continue_btn.clicked.connect(self.action_continue)
        self.yellowcal.continue_btn.clicked.connect(self.action_continue)
        self.segmentcal.continue_btn.clicked.connect(self.action_continue)
        self.snakescal.continue_btn.clicked.connect(self.action_continue)
        self.threshcal.finish_btn.clicked.connect(self.action_finish)

        self.redcal.continue_btn.clicked.connect(self.save_red)
        self.greencal.continue_btn.clicked.connect(self.save_green)
        self.yellowcal.continue_btn.clicked.connect(self.save_yellow)

        # Check for minimal values
        self.redcal.lower_sat_slider.valueChanged.connect(self.check_min)
        self.redcal.lower_val_slider.valueChanged.connect(self.check_min)
        self.redcal.upper_sat_slider.valueChanged.connect(self.check_min)
        self.redcal.upper_val_slider.valueChanged.connect(self.check_min)

        self.greencal.lower_sat_slider.valueChanged.connect(self.check_min)
        self.greencal.lower_val_slider.valueChanged.connect(self.check_min)
        self.greencal.upper_sat_slider.valueChanged.connect(self.check_min)
        self.greencal.upper_val_slider.valueChanged.connect(self.check_min)

        self.yellowcal.lower_sat_slider.valueChanged.connect(self.check_min)
        self.yellowcal.lower_val_slider.valueChanged.connect(self.check_min)
        self.yellowcal.upper_sat_slider.valueChanged.connect(self.check_min)
        self.yellowcal.upper_val_slider.valueChanged.connect(self.check_min)

        self.threshcal.min_thresh_slider.valueChanged.connect(self.check_min)
        self.threshcal.max_thresh_slider.valueChanged.connect(self.check_min)

        # Redraw for each change
        self.redcal.lower_hue_slider.valueChanged.connect(self.redraw)
        self.redcal.lower_sat_slider.valueChanged.connect(self.redraw)
        self.redcal.lower_val_slider.valueChanged.connect(self.redraw)
        self.redcal.upper_hue_slider.valueChanged.connect(self.redraw)
        self.redcal.upper_sat_slider.valueChanged.connect(self.redraw)
        self.redcal.upper_val_slider.valueChanged.connect(self.redraw)

        self.greencal.lower_hue_slider.valueChanged.connect(self.redraw)
        self.greencal.lower_sat_slider.valueChanged.connect(self.redraw)
        self.greencal.lower_val_slider.valueChanged.connect(self.redraw)
        self.greencal.upper_hue_slider.valueChanged.connect(self.redraw)
        self.greencal.upper_sat_slider.valueChanged.connect(self.redraw)
        self.greencal.upper_val_slider.valueChanged.connect(self.redraw)

        self.yellowcal.lower_hue_slider.valueChanged.connect(self.redraw)
        self.yellowcal.lower_sat_slider.valueChanged.connect(self.redraw)
        self.yellowcal.lower_val_slider.valueChanged.connect(self.redraw)
        self.yellowcal.upper_hue_slider.valueChanged.connect(self.redraw)
        self.yellowcal.upper_sat_slider.valueChanged.connect(self.redraw)
        self.yellowcal.upper_val_slider.valueChanged.connect(self.redraw)

        self.segmentcal.param_n_slider.valueChanged.connect(self.redraw)
        self.segmentcal.param_w_slider.valueChanged.connect(self.redraw)
        self.segmentcal.min_diameter_edit.textChanged.connect(self.redraw)

        self.snakescal.apply_btn.clicked.connect(self.redraw)

        self.threshcal.min_thresh_slider.valueChanged.connect(self.redraw)
        self.threshcal.max_thresh_slider.valueChanged.connect(self.redraw)

        self.show()

    def check_min(self):
        try:
            if self.stack.currentWidget().upper_sat_slider.value() < self.stack.currentWidget().lower_sat_slider.value():
                self.stack.currentWidget().upper_sat_slider.setValue(self.stack.currentWidget().lower_sat_slider.value())
            if self.stack.currentWidget().upper_val_slider.value() < self.stack.currentWidget().lower_val_slider.value():
                self.stack.currentWidget().upper_val_slider.setValue(self.stack.currentWidget().lower_val_slider.value())
        except:
            pass
        try:
            if self.stack.currentWidget().max_thresh_slider.value() < self.stack.currentWidget().min_thresh_slider.value():
                self.stack.currentWidget().max_thresh_slider.setValue(self.stack.currentWidget().min_thresh_slider.value())
        except:
            pass

    def redraw(self):
        if self.i in (0, 1, 2):
            if self.stack.currentWidget().lower_hue_slider.value() > self.stack.currentWidget().upper_hue_slider.value():
                boundary = np.array([([int(self.stack.currentWidget().lower_hue_slider.value()/2),
                                       self.stack.currentWidget().lower_sat_slider.value(),
                                       self.stack.currentWidget().lower_val_slider.value()],
                                      [179,
                                       self.stack.currentWidget().upper_sat_slider.value(),
                                       self.stack.currentWidget().upper_val_slider.value()]),
                                     ([0,
                                       self.stack.currentWidget().lower_sat_slider.value(),
                                       self.stack.currentWidget().lower_val_slider.value()],
                                      [int(self.stack.currentWidget().upper_hue_slider.value()/2),
                                       self.stack.currentWidget().upper_sat_slider.value(),
                                       self.stack.currentWidget().upper_val_slider.value()])])

            else:
                boundary = np.array([([int(self.stack.currentWidget().lower_hue_slider.value()/2),
                                       self.stack.currentWidget().lower_sat_slider.value(),
                                       self.stack.currentWidget().lower_val_slider.value()],
                                      [int(self.stack.currentWidget().upper_hue_slider.value()/2),
                                       self.stack.currentWidget().upper_sat_slider.value(),
                                       self.stack.currentWidget().upper_val_slider.value()])])

            t = image_processor.calibrate_color(boundary)
            image = format_image(t)
            self.stack.currentWidget().picture_label.setPixmap(QPixmap.fromImage(image))
            self.stack.currentWidget().picture_label.setScaledContents(True)
            self.stack.currentWidget().picture_label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)


        if self.i == 3:
            image_processor.param_n = self.stack.currentWidget().param_n_slider.value()
            image_processor.param_w = self.stack.currentWidget().param_w_slider.value()
            if self.stack.currentWidget().min_diameter_edit.text() == "":
                image_processor.min_cell_diameter = 0
            else:
                image_processor.min_cell_diameter = int(self.stack.currentWidget().min_diameter_edit.text())

            t = image_processor.calibrate_segmentation()
            image = format_image(t)
            self.stack.currentWidget().picture_label.setPixmap(QPixmap.fromImage(image))
            self.stack.currentWidget().picture_label.setScaledContents(True)
            self.stack.currentWidget().picture_label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

        if self.i == 4:
            image_processor.smoothing = int(self.stack.currentWidget().smoothing_slider.value())
            image_processor.lambda1 = float(1+(self.stack.currentWidget().lambda1_slider.value()/100))
            image_processor.lambda2 = float(1 + (self.stack.currentWidget().lambda2_slider.value() / 100))
            if self.stack.currentWidget().iterations_edit.text() == "":
                image_processor.iterations = 0
            else:
                image_processor.iterations = int(self.stack.currentWidget().iterations_edit.text())

            t = image_processor.calibrate_snake()
            image = format_image(t)
            self.stack.currentWidget().picture_label.setPixmap(QPixmap.fromImage(image))
            self.stack.currentWidget().picture_label.setScaledContents(True)
            self.stack.currentWidget().picture_label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

        if self.i == 5:
            image_processor.tmin = self.stack.currentWidget().min_thresh_slider.value()
            image_processor.tmax = self.stack.currentWidget().max_thresh_slider.value()

            t = image_processor.calibrate_threshold()
            image = format_image(t)
            self.stack.currentWidget().picture_label.setPixmap(QPixmap.fromImage(image))
            self.stack.currentWidget().picture_label.setScaledContents(True)
            self.stack.currentWidget().picture_label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

    def save_red(self):
        return

    def save_green(self):
        return

    def save_yellow(self):
        return


    def action_continue(self):
        self.i = self.i+1
        self.stack.setCurrentIndex(self.i)

    def action_finish(self):
        self.close()


class RedColorCalibrationWindow(cal_red_color_base, cal_red_color_form):
    def __init__(self):
        super(cal_red_color_base, self).__init__()
        self.setupUi(self)

        self.continue_btn.clicked.connect(self.action_continue)

        lower, _ = image_processor.red_boundary[0]
        _, upper = image_processor.red_boundary[1]

        self.lower_hue_slider.setValue(lower[0]*2)
        self.lower_sat_slider.setValue(lower[1])
        self.lower_val_slider.setValue(lower[2])
        self.upper_hue_slider.setValue(upper[0]*2)
        self.upper_sat_slider.setValue(upper[1])
        self.upper_val_slider.setValue(upper[2])

        self.lower_sat_slider.valueChanged.connect(self.check_min)
        self.lower_val_slider.valueChanged.connect(self.check_min)
        self.upper_sat_slider.valueChanged.connect(self.check_min)
        self.upper_val_slider.valueChanged.connect(self.check_min)

        self.lower_hue_slider.valueChanged.connect(self.redraw)
        self.lower_sat_slider.valueChanged.connect(self.redraw)
        self.lower_val_slider.valueChanged.connect(self.redraw)
        self.upper_hue_slider.valueChanged.connect(self.redraw)
        self.upper_sat_slider.valueChanged.connect(self.redraw)
        self.upper_val_slider.valueChanged.connect(self.redraw)

        t = image_processor.calibrate_color(image_processor.red_boundary)
        image = format_image(t)
        self.picture_label.setPixmap(QPixmap.fromImage(image))
        self.picture_label.setScaledContents(True)
        self.picture_label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

    def action_continue(self):
        self.close()

    def check_min(self):
        if self.upper_sat_slider.value() < self.lower_sat_slider.value():
            self.upper_sat_slider.setValue(self.lower_sat_slider.value())
        if self.upper_val_slider.value() < self.lower_val_slider.value():
            self.upper_val_slider.setValue(self.lower_val_slider.value())

    def redraw(self):
        if self.lower_hue_slider.value() > self.upper_hue_slider.value():
            boundary = np.array([([int(self.lower_hue_slider.value() / 2),
                                   self.lower_sat_slider.value(),
                                   self.lower_val_slider.value()],
                                  [179,
                                   self.upper_sat_slider.value(),
                                   self.upper_val_slider.value()]),
                                 ([0,
                                   self.lower_sat_slider.value(),
                                   self.lower_val_slider.value()],
                                  [int(self.upper_hue_slider.value() / 2),
                                   self.upper_sat_slider.value(),
                                   self.upper_val_slider.value()])])

        else:
            boundary = np.array([([int(self.lower_hue_slider.value() / 2),
                                   self.lower_sat_slider.value(),
                                   self.lower_val_slider.value()],
                                  [int(self.upper_hue_slider.value() / 2),
                                   self.upper_sat_slider.value(),
                                   self.upper_val_slider.value()])])

        t = image_processor.calibrate_color(boundary)
        image = format_image(t)
        self.picture_label.setPixmap(QPixmap.fromImage(image))
        self.picture_label.setScaledContents(True)
        self.picture_label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)


class GreenColorCalibrationWindow(cal_green_color_base, cal_green_color_form):
    def __init__(self):
        super(cal_green_color_base, self).__init__()
        self.setupUi(self)

        self.continue_btn.clicked.connect(self.action_continue)

        lower, upper = image_processor.green_boundary[0]

        self.lower_hue_slider.setValue(lower[0]*2)
        self.lower_sat_slider.setValue(lower[1])
        self.lower_val_slider.setValue(lower[2])
        self.upper_hue_slider.setValue(upper[0]*2)
        self.upper_sat_slider.setValue(upper[1])
        self.upper_val_slider.setValue(upper[2])

        self.lower_sat_slider.valueChanged.connect(self.check_min)
        self.lower_val_slider.valueChanged.connect(self.check_min)
        self.upper_sat_slider.valueChanged.connect(self.check_min)
        self.upper_val_slider.valueChanged.connect(self.check_min)

        self.lower_hue_slider.valueChanged.connect(self.redraw)
        self.lower_sat_slider.valueChanged.connect(self.redraw)
        self.lower_val_slider.valueChanged.connect(self.redraw)
        self.upper_hue_slider.valueChanged.connect(self.redraw)
        self.upper_sat_slider.valueChanged.connect(self.redraw)
        self.upper_val_slider.valueChanged.connect(self.redraw)

        t = image_processor.calibrate_color(image_processor.green_boundary)
        image = format_image(t)
        self.picture_label.setPixmap(QPixmap.fromImage(image))
        self.picture_label.setScaledContents(True)
        self.picture_label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

    def action_continue(self):
        self.close()

    def check_min(self):
        if self.upper_sat_slider.value() < self.lower_sat_slider.value():
            self.upper_sat_slider.setValue(self.lower_sat_slider.value())
        if self.upper_val_slider.value() < self.lower_val_slider.value():
            self.upper_val_slider.setValue(self.lower_val_slider.value())

    def redraw(self):
        if self.lower_hue_slider.value() > self.upper_hue_slider.value():
            boundary = np.array([([int(self.lower_hue_slider.value() / 2),
                                   self.lower_sat_slider.value(),
                                   self.lower_val_slider.value()],
                                  [179,
                                   self.upper_sat_slider.value(),
                                   self.upper_val_slider.value()]),
                                 ([0,
                                   self.lower_sat_slider.value(),
                                   self.lower_val_slider.value()],
                                  [int(self.upper_hue_slider.value() / 2),
                                   self.upper_sat_slider.value(),
                                   self.upper_val_slider.value()])])

        else:
            boundary = np.array([([int(self.lower_hue_slider.value() / 2),
                                   self.lower_sat_slider.value(),
                                   self.lower_val_slider.value()],
                                  [int(self.upper_hue_slider.value() / 2),
                                   self.upper_sat_slider.value(),
                                   self.upper_val_slider.value()])])

        t = image_processor.calibrate_color(boundary)
        image = format_image(t)
        self.picture_label.setPixmap(QPixmap.fromImage(image))
        self.picture_label.setScaledContents(True)
        self.picture_label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)


class YellowColorCalibrationWindow(cal_yellow_color_base, cal_yelow_color_form):
    def __init__(self):
        super(cal_yellow_color_base, self).__init__()
        self.setupUi(self)

        self.continue_btn.clicked.connect(self.action_continue)

        lower, upper = image_processor.yellow_boundary[0]

        self.lower_hue_slider.setValue(lower[0]*2)
        self.lower_sat_slider.setValue(lower[1])
        self.lower_val_slider.setValue(lower[2])
        self.upper_hue_slider.setValue(upper[0]*2)
        self.upper_sat_slider.setValue(upper[1])
        self.upper_val_slider.setValue(upper[2])

        self.lower_sat_slider.valueChanged.connect(self.check_min)
        self.lower_val_slider.valueChanged.connect(self.check_min)
        self.upper_sat_slider.valueChanged.connect(self.check_min)
        self.upper_val_slider.valueChanged.connect(self.check_min)

        self.lower_hue_slider.valueChanged.connect(self.redraw)
        self.lower_sat_slider.valueChanged.connect(self.redraw)
        self.lower_val_slider.valueChanged.connect(self.redraw)
        self.upper_hue_slider.valueChanged.connect(self.redraw)
        self.upper_sat_slider.valueChanged.connect(self.redraw)
        self.upper_val_slider.valueChanged.connect(self.redraw)

        t = image_processor.calibrate_color(image_processor.yellow_boundary)
        image = format_image(t)
        self.picture_label.setPixmap(QPixmap.fromImage(image))
        self.picture_label.setScaledContents(True)
        self.picture_label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

    def action_continue(self):
        self.close()

    def check_min(self):
        if self.upper_sat_slider.value() < self.lower_sat_slider.value():
            self.upper_sat_slider.setValue(self.lower_sat_slider.value())
        if self.upper_val_slider.value() < self.lower_val_slider.value():
            self.upper_val_slider.setValue(self.lower_val_slider.value())

    def redraw(self):
        if self.lower_hue_slider.value() > self.upper_hue_slider.value():
            boundary = np.array([([int(self.lower_hue_slider.value() / 2),
                                   self.lower_sat_slider.value(),
                                   self.lower_val_slider.value()],
                                  [179,
                                   self.upper_sat_slider.value(),
                                   self.upper_val_slider.value()]),
                                 ([0,
                                   self.lower_sat_slider.value(),
                                   self.lower_val_slider.value()],
                                  [int(self.upper_hue_slider.value() / 2),
                                   self.upper_sat_slider.value(),
                                   self.upper_val_slider.value()])])

        else:
            boundary = np.array([([int(self.lower_hue_slider.value() / 2),
                                   self.lower_sat_slider.value(),
                                   self.lower_val_slider.value()],
                                  [int(self.upper_hue_slider.value() / 2),
                                   self.upper_sat_slider.value(),
                                   self.upper_val_slider.value()])])

        t = image_processor.calibrate_color(boundary)
        image = format_image(t)
        self.picture_label.setPixmap(QPixmap.fromImage(image))
        self.picture_label.setScaledContents(True)
        self.picture_label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)


class SegmentationCalibrationWindow(cal_segmentation_base, cal_segmentation_form):
    def __init__(self):
        super(cal_segmentation_base, self).__init__()
        self.setupUi(self)

        self.continue_btn.clicked.connect(self.action_continue)

        self.min_diameter_edit.setValidator(QIntValidator())

        self.param_n_slider.setValue(image_processor.param_n)
        self.param_w_slider.setValue(image_processor.param_w)
        self.min_diameter_edit.setText(str(image_processor.min_cell_diameter))

        self.param_n_slider.valueChanged.connect(self.redraw)
        self.param_w_slider.valueChanged.connect(self.redraw)
        self.min_diameter_edit.textChanged.connect(self.redraw)

        t = image_processor.calibrate_segmentation()
        image = format_image(t)
        self.picture_label.setPixmap(QPixmap.fromImage(image))
        self.picture_label.setScaledContents(True)
        self.picture_label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

    def action_continue(self):
        self.close()

    def redraw(self):
        image_processor.param_n = self.param_n_slider.value()
        image_processor.param_w = self.param_w_slider.value()
        if self.min_diameter_edit.text() == "":
            image_processor.min_cell_diameter = 0
        else:
            image_processor.min_cell_diameter = int(self.min_diameter_edit.text())

        t = image_processor.calibrate_segmentation()
        image = format_image(t)
        self.picture_label.setPixmap(QPixmap.fromImage(image))
        self.picture_label.setScaledContents(True)
        self.picture_label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)


class SnakesCalibrationWindow(cal_snakes_base, cal_snakes_form):
    def __init__(self):
        super(cal_snakes_base, self).__init__()
        self.setupUi(self)

        self.continue_btn.clicked.connect(self.action_continue)
        self.apply_btn.clicked.connect(self.redraw)

        self.iterations_edit.setValidator(QIntValidator())

        self.smoothing_slider.setValue(image_processor.smoothing)
        self.lambda1_slider.setValue(image_processor.lambda1)
        self.lambda2_slider.setValue(image_processor.lambda2)
        self.iterations_edit.setText(str(image_processor.iterations))

        self.redraw_label.setText(' ')

    def action_continue(self):
        self.close()

    def redraw(self):
        image_processor.smoothing = int(self.smoothing_slider.value())
        image_processor.lambda1 = float(1 + (self.lambda1_slider.value() / 100))
        image_processor.lambda2 = float(1 + (self.lambda2_slider.value() / 100))
        if self.iterations_edit.text() == "":
            image_processor.iterations = 0
        else:
            image_processor.iterations = int(self.iterations_edit.text())

        t = image_processor.calibrate_snake()
        image = format_image(t)
        self.picture_label.setPixmap(QPixmap.fromImage(image))
        self.picture_label.setScaledContents(True)
        self.picture_label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)


class ThresholdCalibrationWindow(cal_threshold_base, cal_threshold_form):
    def __init__(self):
        super(cal_threshold_base, self).__init__()
        self.setupUi(self)

        self.finish_btn.clicked.connect(self.action_continue)

        self.min_thresh_slider.setValue(int(image_processor.tmin-1)*100)
        self.max_thresh_slider.setValue(int(image_processor.tmax-1)*100)

        self.min_thresh_slider.valueChanged.connect(self.check_min)
        self.max_thresh_slider.valueChanged.connect(self.check_min)

        self.min_thresh_slider.valueChanged.connect(self.redraw)
        self.max_thresh_slider.valueChanged.connect(self.redraw)

        t = image_processor.calibrate_threshold()
        image = format_image(t)
        self.picture_label.setPixmap(QPixmap.fromImage(image))
        self.picture_label.setScaledContents(True)
        self.picture_label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

    def action_continue(self):
        self.close()

    def check_min(self):
        if self.max_thresh_slider.value() < self.min_thresh_slider.value():
            self.max_thresh_slider.setValue(self.min_thresh_slider.value())

    def redraw(self):
        image_processor.tmin = self.min_thresh_slider.value()
        image_processor.tmax = self.max_thresh_slider.value()

        t = image_processor.calibrate_threshold()
        image = format_image(t)
        self.picture_label.setPixmap(QPixmap.fromImage(image))
        self.picture_label.setScaledContents(True)
        self.picture_label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)


class OutputProtocolEditWindow(output_protocol_base, output_protocol_form):
    def __init__(self):
        super(output_protocol_base, self).__init__()
        self.setupUi(self)

        self.save_btn.clicked.connect(self.action_save)

    def action_save(self):
        return


class ParameterWindow(parameters_base, parameters_form):
    def __init__(self):
        super(parameters_base, self).__init__()
        self.setupUi(self)

        self.n_edit_val.setText(str(image_processor.param_n))
        self.w_edit_val.setText(str(image_processor.param_w))
        self.min_diameter_edit_val.setText(str(image_processor.min_cell_diameter))
        self.smoothing_edit_val.setText(str(image_processor.smoothing))
        self.lambda1_edit_val.setText(str(image_processor.lambda1))
        self.lambda2_edit_val.setText(str(image_processor.lambda2))
        self.min_thresh_edit_val.setText(str(image_processor.tmin))
        self.max_thresh_edit_val.setText(str(image_processor.tmax))

        self.save_btn.clicked.connect(self.action_save)



    def action_save(self):
        image_processor.param_n = int(self.n_edit_val.text())
        image_processor.param_w = int(self.w_edit_val.text())
        image_processor.min_cell_diameter = int(self.min_diameter_edit_val.text())
        image_processor.smoothing = int(self.smoothing_edit_val.text())
        image_processor.lambda1 = float(self.lambda1_edit_val.text())
        image_processor.lambda2 = float(self.lambda2_edit_val.text())
        image_processor.tmin = int(self.min_thresh_edit_val.text())
        image_processor.tmax = int(self.max_thresh_edit_val.text())
        image_processor.analyze()
        self.close()
        return




if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())
