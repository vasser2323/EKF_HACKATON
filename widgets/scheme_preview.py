import os
import cv2
import fitz
import tempfile
import numpy as np
from icecream import ic
import utilities
import yaml

from PyQt5.QtWidgets import (
    QGraphicsView,
    QFileDialog,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QComboBox,
    QDialog,
    QLabel,
    QVBoxLayout,
    QLineEdit,
    QHBoxLayout,
    QPushButton,
    QGraphicsTextItem,
    QGraphicsRectItem,
    
)
from PyQt5.QtCore import QPointF, QRectF, pyqtSignal
from PyQt5.QtGui import (
    QPen,
    QFont,
    QColor,
    QPixmap,
    QPainter,
    QPolygonF,
    QImage,
    QBrush,

)
from PyQt5.Qt import Qt
import config_load


class ShapeDialog(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)

        with open("config.yaml") as config_file:
            config = yaml.safe_load(config_file)
            self.classes = config["data"]["classes_path"]

        self.setWindowTitle("Enter Shape Details")

        self.layout = QVBoxLayout(self)

        self.name_label = QLabel("Shape Name:", self)
        self.layout.addWidget(self.name_label)

        self.name_edit = QLineEdit(self)
        self.layout.addWidget(self.name_edit)

        self.class_label = QLabel("Shape Class:", self)
        self.layout.addWidget(self.class_label)

        
        self.class_combo: QComboBox = QComboBox(self)
        self.class_combo.addItems(
            utilities.read_classes(os.path.join(self.classes))
        ) 
        self.class_combo.currentIndexChanged.connect(self.update_text_input)
        self.layout.addWidget(self.class_combo)

        self.button_box = QHBoxLayout(self)
        self.ok_button = QPushButton("OK", self)
        self.cancel_button = QPushButton("Cancel", self)
        self.button_box.addWidget(self.ok_button)
        self.button_box.addWidget(self.cancel_button)
        self.layout.addLayout(self.button_box)

        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

    def update_text_input(self, index):
        self.name_edit.setText(self.class_combo.currentText())
    
    def get_details(self):
        return self.name_edit.text().upper()


class SchemePreview(QGraphicsView):

    scheme_image: np.ndarray
    scheme_image_pixmap: QPixmap
    current_rect_item = None
    current_text_item = None
    draw_mode = False

    last_clicked_coords: QPointF

    bounding_box_added: pyqtSignal = pyqtSignal(str, tuple)
    bounding_box_deleted: pyqtSignal = pyqtSignal(QPointF)
    scheme_selection_requested: pyqtSignal = pyqtSignal()

    def __init__(self, scene):
        super().__init__(scene)
        self.scene = QGraphicsScene(self)
        self.scheme_image = None
        self.scheme_image_pixmap = None

        # Zooming and moving the image
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setRenderHint(QPainter.TextAntialiasing)

        # Boxes selection
        self.setMouseTracking(True)
        self.current_rect_item = None
        self.current_text_item = None
        self.draw_mode = False
        self.origin = QPointF()
        self.new_boxes = [] #vs
        self.last_clicked_coords = None

        self.min_scale = 0.1
        self.max_scale = 10

        self.selected_item = None
        self.highlight_item = None
    
    def display_image(self):
        if self.scheme_image_pixmap is None:
            return
        
        pixmap_item = QGraphicsPixmapItem(self.scheme_image_pixmap)
        self.scene.addItem(pixmap_item)
        self.setScene(self.scene)
    
    # Load scheme image file
    def load_image(self, file_path):
        # Clear previous contents before loading new image
        self.scene.clear()
        self.current_rect_item = None

        if file_path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".xpm")):
            self.scheme_image = cv2.imread(file_path, cv2.IMREAD_COLOR)
            self.scheme_image_pixmap = QPixmap(file_path)
            self.display_image()

        elif file_path.lower().endswith(".pdf"):
            # Convert first page of PDF to image
            doc = fitz.open(file_path)
            pdf_page = doc.load_page(0)
            pixmap = pdf_page.get_pixmap()
            img = QImage(
                pixmap.samples,
                pixmap.width,
                pixmap.height,
                pixmap.stride,
                QImage.Format_RGB888,
            )
            doc.close()

            # Save image to jpg file
            temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            img.save(temp_file.name)
            temp_file.close()

            # Load image from saved file
            self.load_image(temp_file.name)

    # Display YOLO predictions on the image
    def visualize_predictions(
        self,
        detection_results: list,
        shkaf_detection_results: list,
    ):

        def get_colors_for_classes(class_list: list[str]) -> dict[str, str]:
            colors = []
            color_list = [
                "red",
                "green",
                "blue",
                "yellow",
                "orange",
                "purple",
                "cyan",
                "magenta",
            ]

            for index, class_name in enumerate(class_list):
                color_index = index % len(color_list)
                color = QColor(color_list[color_index])
                colors.append((class_name, color))

            class_colors = dict(colors)
            return class_colors

        # Clear previous contours
        self.scene.clear()
        self.current_rect_item = None
        self.display_image()

        # Set color for shkaf
        shkaf_color = QColor("black")

        classes_list = []
        with open("config.yaml") as config_file:
            config = yaml.safe_load(config_file)
            classes_path = config["data"]["classes_path"]
            classes_list=utilities.read_classes(classes_path)
        
        # convert from indices to text
        class_names: list[str] = [
            classes_list[class_index] for class_index in detection_results["class_indices"]
        ]

        # get non-text classes
        unique_class_names: list[str] = [
            class_name
            for class_name in set(classes_list)
            if not class_name.endswith("TEXT")
        ]
        class_colors: dict[str, str] = get_colors_for_classes(unique_class_names)

        for box, class_name in zip(detection_results["boxes"], class_names):
            x, y, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]

            # Set bounding box color
            if not class_name.startswith("SHKAF"):
                if class_name.endswith("TEXT"):
                    color: QColor = class_colors[class_name.split("_")[0]]
                    color.setAlphaF(0.5)
                else:
                    color: QColor = class_colors[class_name]
                    color.setAlphaF(1.0)
            else:
                color = shkaf_color
            pen = QPen(color, 2)

            # Draw bounding box
            self.scene.addRect(QRectF(x, y, w, h), pen)

            # Display element class
            text = QGraphicsTextItem(class_name)
            color.setAlphaF(1.0)
            text.setDefaultTextColor(color)
            text.setFont(QFont("Arial", 8))
            text_height = text.boundingRect().height()
            text.setPos(x, y - text_height)
            self.scene.addItem(text)

        for box in shkaf_detection_results["boxes"]:
            x, y, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]

            # Set color
            pen = QPen(shkaf_color, 5)

            # Draw bounding box
            self.scene.addRect(QRectF(x, y, w, h), pen)

            # Display element class
            text = QGraphicsTextItem("SHKAF")
            text.setDefaultTextColor(shkaf_color)
            text.setFont(QFont("Arial", 12))
            text_height = text.boundingRect().height()
            text.setPos(x, y - text_height)
            self.scene.addItem(text)

    # Scale the image
    def wheelEvent(self, event):
        zoomFactor = 1.1 if event.angleDelta().y() > 0 else 0.9
        old_pos = self.mapToScene(event.pos())
        self.scale(zoomFactor, zoomFactor)
        new_pos = self.mapToScene(event.pos())
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())

        # Clamp zoom level
        current_scale = self.transform().m11()
        if current_scale < self.min_scale:
            self.scale(self.min_scale / current_scale, self.min_scale / current_scale)
        elif current_scale > self.max_scale:
            self.scale(self.max_scale / current_scale, self.max_scale / current_scale)

    def mousePressEvent(self, event):
        # If scheme image is already selected
        if self.scheme_image is not None:
            # Move image
            if event.button() == Qt.LeftButton:
                self.setDragMode(QGraphicsView.ScrollHandDrag)

            # Select bounding box
            elif event.button() == Qt.RightButton:

                self.origin = self.mapToScene(event.pos())

                if self.current_rect_item is None:
                    pen = QPen(Qt.green, 2)
                    self.current_rect_item = self.scene.addRect(
                        QRectF(self.origin, self.origin), pen
                    )
                else:
                    if self.current_rect_item in self.scene.items():
                        self.current_rect_item.setRect(QRectF(self.origin, self.origin))

                self.draw_mode = True

                self.clear_selection()

        # Select an image file or PDF file
        else:
            if event.button() == Qt.LeftButton:
                self.scheme_selection_requested.emit()

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.draw_mode:
            end_pos = self.mapToScene(event.pos())
            self.current_rect_item.setRect(QRectF(self.origin, end_pos).normalized())
        QGraphicsView.mouseMoveEvent(self, event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.setDragMode(QGraphicsView.NoDrag)

        elif event.button() == Qt.RightButton:
            self.draw_mode = False

            start_pos = self.current_rect_item.rect().topLeft()
            end_pos = self.current_rect_item.rect().bottomRight()

            # Add contour
            min_size = 10
            if (
                abs(end_pos.x() - start_pos.x()) >= min_size
                or abs(end_pos.y() - start_pos.y()) >= min_size
            ):
                self.drawRectangle(start_pos, end_pos)
                self.last_clicked_coords=None
            
            # Select contour
            else:
                self.last_clicked_coords=start_pos
                self.select_item(self.last_clicked_coords)
                ic(self.last_clicked_coords)

        super().mouseReleaseEvent(event)

    def clear_selection(self):
        if self.highlight_item and self.highlight_item in self.scene.items():
            self.scene.removeItem(self.highlight_item)
            self.highlight_item = None
        self.selected_item = None


    def select_item(self, pos):
        self.clear_selection()

        for item in self.scene.items(pos):
            if isinstance(item, QGraphicsRectItem):
                self.selected_item = item
                self.highlight_selected_item()
                break

    def highlight_selected_item(self):
        if self.selected_item:
            highlight_color = QColor(255, 255, 0, 100)  # Полупрозрачный желтый
            self.highlight_item = self.scene.addRect(self.selected_item.rect(), QPen(Qt.NoPen), QBrush(highlight_color))
            self.highlight_item.setZValue(self.selected_item.zValue() - 1)  # Размещаем подсветку под элементом


    def is_point_inside_box(self, point, box):
        return (box[0] <= point.x() <= box[2]) and (box[1] <= point.y() <= box[3])

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete or event.key() == Qt.Key_Backspace:
            if self.last_clicked_coords is not None:
                self.delete_bounding_box(self.last_clicked_coords)
        else:
            super().keyPressEvent(event)

    def delete_bounding_box(self, point):

        # items_to_remove = []

        # for item in self.scene.items():

        #     if isinstance(item, QGraphicsRectItem):
        #         rect = item.rect()
        #         if self.is_point_inside_box(point, (rect.left(), rect.top(), rect.right(), rect.bottom())):
        #             items_to_remove.append(item)

        #     elif isinstance(item, QGraphicsTextItem):
        #         if self.is_point_inside_box(point, (item.x(), item.y(), item.x() + item.boundingRect().width(), item.y() + item.boundingRect().height())):
        #             items_to_remove.append(item)
        
        # for item in items_to_remove:
        #     self.scene.removeItem(item)
        
        self.bounding_box_deleted.emit(point)

    def drawRectangle(self, start_pos, end_pos):

        dialog = ShapeDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            
            rect = self.create_rect(start_pos, end_pos)
            pen = QPen(Qt.red, 2)
            self.scene.addPolygon(rect, pen)

            shape_name = dialog.get_details()
            
            if shape_name:
                text_item = self.scene.addText(f"{shape_name}")
                text_item.setPos(start_pos)
                text_item.setDefaultTextColor(Qt.blue)

            new_box = (shape_name, (
                int(start_pos.x()),
                int(start_pos.y()),
                int(end_pos.x()),
                int(end_pos.y())
            ))
            self.new_boxes.append(new_box) ##VS

            self.bounding_box_added.emit(shape_name, (
                int(start_pos.x()),
                int(start_pos.y()),
                int(end_pos.x()),
                int(end_pos.y())
            ))

    def create_rect(self, start_point, end_point):
        x1 = start_point.x()
        y1 = start_point.y()
        x2 = end_point.x()
        y2 = end_point.y()

        rect = QPolygonF()
        rect.append(QPointF(x1, y1))
        rect.append(QPointF(x2, y1))
        rect.append(QPointF(x2, y2))
        rect.append(QPointF(x1, y2))

        return rect
