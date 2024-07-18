import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import sys
import yaml
import numpy as np
from icecream import ic
import utilities
from paddleocr import PaddleOCR

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QFileDialog,
    QSizePolicy,
    QAction,
    QFileDialog,
    QRadioButton,
    QDialog,
    QMessageBox,
)
from PyQt5.QtCore import QThread, pyqtSignal, QPointF, QUrl
from PyQt5.Qt import Qt
from PyQt5.QtGui import QDesktopServices


from widgets.stylized_widget import StylizedWidget
from widgets.plan_preview_table import PlanPreviewTable
from widgets.scheme_preview import SchemePreview

from ultralytics import YOLO

import paddle

class PlanEstimator(QMainWindow, StylizedWidget):

    model: YOLO
    shkaf_model: YOLO
    ocr: PaddleOCR

    classes_list: list[str] = None
    model_file_path: str = None
    shkaf_model_file_path: str = None
    dataframe_path: str = None
    abbreviations_path: str = None
    user_guide_path: str = None

    last_analyze_results: list
    schema_analyzer_thread: QThread
    detection_results: list
    shkaf_detection_results: list

    def __init__(self, parent=None):
        super().__init__(parent)

        # Load settings from config file
        with open("config.yaml") as config_file:
            config = yaml.safe_load(config_file)
            self.model_file_path = config["model"]["model_file_path"]
            self.shkaf_model_file_path = config["model"]["shkaf_model_file_path"]
            self.dataframe_path = config["data"]["dataframe_path"]
            self.abbreviations_path = config["data"]["abbreviations_path"]
            self.user_guide_path = config["data"]["user_guide_path"]

            classes_path = config["data"]["classes_path"]
            self.classes_list = utilities.read_classes(classes_path)

        ic(self.model_file_path)

        self.ocr = None
        self.model = None
        self.detection_results = None
        self.selected_file_path = None
        self.schema_analyzer_thread = None
        self.last_analyze_results = None
        

        # Initialize UI components
        self.initialize_UI()

        # Show the main window
        self.show()
        self.center()
        self.setMinimumSize(950, 350)

        # Set color theme
        self.set_styles()

    def initialize_UI(self):
        # Create main layout element
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.main_layout = QHBoxLayout(self.main_widget)

        self.scheme_layout = QVBoxLayout(self)
        self.scheme_layout.setSpacing(10)
        self.main_layout.addLayout(self.scheme_layout)
        
        # Add image preview
        self.scheme_preview = SchemePreview(self.main_widget)
        self.scheme_preview.scheme_selection_requested.connect(self.select_scheme_image)
        self.scheme_preview.bounding_box_added.connect(self.on_contour_added)
        self.scheme_preview.bounding_box_deleted.connect(self.on_contour_deleted)
        self.scheme_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.scheme_preview.setAlignment(Qt.AlignCenter)
        self.scheme_layout.addWidget(self.scheme_preview)

        # Add button for processing scheme image
        self.analyze_button = QPushButton("Analyze", self)
        self.analyze_button.clicked.connect(self.analyze_scheme)
        self.scheme_layout.addWidget(self.analyze_button)

        # Add table view
        self.table_view = PlanPreviewTable(self.main_widget)
        self.table_view.setMinimumWidth(370)
        self.main_layout.addWidget(self.table_view)
        self.scheme_preview.bounding_box_deleted.connect(self.on_contour_deleted)

        # Create top menu bar
        self._createMenuBar()

    def _createMenuBar(self):
        menuBar = self.menuBar()

        # File menu
        fileMenu = menuBar.addMenu("File")

        openAction = QAction("Open scheme image", self)
        openAction.setShortcut("Ctrl+O")
        openAction.triggered.connect(self.select_scheme_image)
        fileMenu.addAction(openAction)

        saveAction = QAction("Export...", self)
        saveAction.setShortcut("Ctrl+S")
        saveAction.triggered.connect(self.export_estimate)
        fileMenu.addAction(saveAction)
        
        fileMenu.addSeparator()

        exitAction = QAction("Quit", self)
        exitAction.setShortcut("Alt+Q")
        exitAction.triggered.connect(self.close)
        fileMenu.addAction(exitAction)

        # Help menu
        helpMenu = menuBar.addMenu("Help")
        self.aboutAction = QAction("About", self)
        self.aboutAction.setShortcut("Ctrl+H")
        self.aboutAction.triggered.connect(self.open_user_guide)
        helpMenu.addAction(self.aboutAction)

    def showEvent(self, event):
        super().showEvent(event)

        # Load model file after application startup
        ic("Loading models...")
        self.model = utilities.load_model(self.model_file_path)
        self.shkaf_model = utilities.load_model(self.shkaf_model_file_path)
        ic("Models are loaded successfully")

        # Initialize ocr
        ic("Initializing OCR...")
        self.ocr = PaddleOCR(lang="en", use_angle_cls=True, show_log=False)
        ic("OCR initialized successfully")

    def open_user_guide(self):
            if os.path.exists(self.user_guide_path):
                QDesktopServices.openUrl(QUrl.fromLocalFile(self.user_guide_path))
            else:
                QMessageBox.warning(self, "Error", f"User guide file not found: {self.user_guide_path}")

    def create_export_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Export Format")
        layout = QVBoxLayout(dialog)

        excel_button = QRadioButton("Excel", dialog)
        csv_button = QRadioButton("CSV", dialog)
        both_button = QRadioButton("Both", dialog)

        layout.addWidget(excel_button)
        layout.addWidget(csv_button)
        layout.addWidget(both_button)

        excel_button.setChecked(True)

        ok_button = QPushButton("OK", dialog)
        ok_button.clicked.connect(dialog.accept)
        layout.addWidget(ok_button)

        dialog.setLayout(layout)

        if dialog.exec() == QDialog.Accepted:
            if excel_button.isChecked():
                return "excel"
            elif csv_button.isChecked():
                return "csv"
            elif both_button.isChecked():
                return "both"
        return None
        
    def select_scheme_image(self):
        # Ask the user to select scheme image
        file_dialog = QFileDialog()
        file_dialog.setNameFilter(
            "Images (*.png *.jpg *.jpeg *.bmp *.xpm);;PDF Files (*.pdf)"
        )
        file_dialog.setViewMode(QFileDialog.Detail)

        # Load image from selected file
        if file_dialog.exec_():
            
            # Display scheme image
            file_path = file_dialog.selectedFiles()[0]
            self.scheme_preview.load_image(file_path)

            self.selected_file_path = file_path

            # Process image file
            self.detection_results = None
            self.process_selected_image_file()

    def process_selected_image_file(self):
        if self.model is not None:
            # Make YOLO predictions
            self.detection_results = utilities.get_predictions(
                self.model, self.scheme_preview.scheme_image
            )

            # Make YOLO predictions of shkafs
            self.shkaf_detection_results = utilities.get_predictions(
                self.shkaf_model, self.scheme_preview.scheme_image
            )

            # Display predicted classes on the scheme image
            self.scheme_preview.visualize_predictions(
                self.detection_results, self.shkaf_detection_results
            )
            

    def on_contour_added(self, object_name: str, contour_coords: tuple):

        if object_name not in self.classes_list:
            return
        
        bouding_box = list(contour_coords)
        class_index = self.classes_list.index(object_name)
        confidence_score = 1

        if object_name != "SHKAF":
            self.detection_results["boxes"].append(bouding_box)
            self.detection_results["class_indices"].append(class_index)
            self.detection_results["confidence_scores"].append(confidence_score)
        else:
            self.shkaf_detection_results["boxes"].append(bouding_box)
            self.shkaf_detection_results["class_indices"].append(class_index)
            self.shkaf_detection_results["confidence_scores"].append(confidence_score)

    def on_contour_deleted(self, point: QPointF):

        if self.detection_results is None:
            return

        for i, box in enumerate(self.detection_results["boxes"]):
            if self.scheme_preview.is_point_inside_box(point, box):
                self.detection_results["boxes"].pop(i)
                self.detection_results["class_indices"].pop(i)
                self.detection_results["confidence_scores"].pop(i)
                break
        
        for i, box in enumerate(self.shkaf_detection_results["boxes"]):
            if self.scheme_preview.is_point_inside_box(point, box):
                self.shkaf_detection_results["boxes"].pop(i)
                self.shkaf_detection_results["class_indices"].pop(i)
                self.shkaf_detection_results["confidence_scores"].pop(i)
                break

        # Обновите визуализацию после удаления
        self.scheme_preview.visualize_predictions(self.detection_results, self.shkaf_detection_results)
        
       
    class AnalyzeSchemaThread(QThread):
        finished: pyqtSignal = pyqtSignal(list)

        def __init__(self, ocr: PaddleOCR, scheme_image: np.ndarray, detection_results: list, shkaf_detection_results:list, dataframe_path:str, abbreviations_path:str,):
            super().__init__()
            
            self.ocr = ocr
            self.scheme_image = scheme_image
            self.detection_results = detection_results
            self.shkaf_detection_results = shkaf_detection_results
            self.dataframe_path = dataframe_path
            self.abbreviations_path = abbreviations_path

        def run(self):
            if self.detection_results is None or self.shkaf_detection_results is None:
                ic("Cannot proceed to export, detections are None")
                return

            # ----------------------------------------------------------------
            # Bind text boxes to nearly objects
            objects_info: list[tuple[str, tuple, tuple]] = utilities.bind_text_boxes(
                self.ocr,
                self.scheme_image,
                self.detection_results,
                self.shkaf_detection_results,
            )

            ic(objects_info)

            object_names_and_text_coords: list[tuple[str, tuple]] = []
            for object_info in objects_info:
                object_names_and_text_coords.append(
                    (object_info[0], object_info[1], object_info[3])
                )


            # ----------------------------------------------------------------
            # Recognize text for each object
            object_names_and_texts: list[tuple[str, str]] = utilities.recognize_text(
                self.ocr, self.scheme_image, object_names_and_text_coords
            )
            ic(object_names_and_texts)

            # ----------------------------------------------------------------
            # Prepare dataframe
            df_n = utilities.prepare_dataframe(self.dataframe_path)

            # ----------------------------------------------------------------
            # Read abbreviations
            abbreviations = utilities.read_dict_from_txt(self.abbreviations_path)
            abbreviations_list = utilities.read_list_from_txt(self.abbreviations_path)

            # ----------------------------------------------------------------
            # Apply fuzzy matching
            find_el = utilities.process_search_list(
                search_list=object_names_and_texts,
                df=df_n,
                abbreviations=abbreviations,
                threshold=40,
            )

            self.finished.emit(find_el)
    
    def analyze_scheme(self):
        if self.scheme_preview.scheme_image is None:
            return
        
        # Disable button for the time of scheme analysis
        self.analyze_button.setEnabled(False)
        self.analyze_button.setStyleSheet("color: #c0c0c0")
        
        # Create separate thread for scheme analysis
        self.schema_analyzer_thread = PlanEstimator.AnalyzeSchemaThread(
            self.ocr,
            self.scheme_preview.scheme_image,
            self.detection_results,
            self.shkaf_detection_results,
            self.dataframe_path,
            self.abbreviations_path,
        )
        self.schema_analyzer_thread.finished.connect(self.on_table_analyzed)
        
        # Start the thread
        self.schema_analyzer_thread.start()

    def on_table_analyzed(self, results: list):
        self.last_analyze_results = results

        self.analyze_button.setEnabled(True)
        self.analyze_button.setStyleSheet(f"color: {self.fg_color}")

        # Calculate objects count for all found articles
        article_occurences: dict[str, int] = {}
        article_params: dict[str, tuple] = {}  # article: (shkaf_name, nomenclature, price)

        for result in sorted(results, key=lambda x: x[0]):  # sort by shraf name
            if result[2]:
                shkaf_name, search_string, (best_match, score, corresponding_row) = result
                article, nomenclature, cost = corresponding_row

                if article not in article_occurences:
                    article_occurences[article] = 1
                else:
                    article_occurences[article] += 1

                if article not in article_params:
                    article_params[article] = (shkaf_name, nomenclature, cost)
        
        # Update table with objects count
        self.table_view.update_table(article_occurences, article_params)

        # Exit thread
        self.schema_analyzer_thread.quit()
        self.schema_analyzer_thread.wait()
    
    # ----------------------------------------------------------------
    # Export to xlsx
    def export_estimate(self):
        if self.last_analyze_results is None:
            return

        output_format = self.create_export_dialog()
    
        output_file_name = os.path.splitext(os.path.basename(self.selected_file_path))[0]
        utilities.export_table(self.table_view.get_table_data(), output_file_name, output_format)
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PlanEstimator()
    sys.exit(app.exec_())
