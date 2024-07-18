import sys
from icecream import ic
from PyQt5.QtWidgets import (
    QGraphicsView,
    QApplication,
    QFileDialog,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QRubberBand,
)
from PyQt5.QtCore import (
    QPointF,
    QRect,
    QPoint,
    QSize,
)
from PyQt5.QtGui import (
    QPen,
    QColor,
    QBrush,
    QPixmap,
    QPainter,
    QPolygonF,
    QPalette,
)
from PyQt5.Qt import Qt


class SchemePreview(QGraphicsView):
    scheme_image: QPixmap

    rubber_bands = []
    current_rubber_band = None
    box_origin_point = None
    draw_mode = False

    def __init__(self, scene):
        super().__init__(scene)

        self.scheme_image = None
        self.scene = QGraphicsScene(self)

        # Zooming and moving the image
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setRenderHint(QPainter.TextAntialiasing)

        # Boxes selection
        self.setMouseTracking(True)
        self.rubber_bands = []
        self.current_rubber_band = None
        self.draw_mode = False
        self.origin = QPoint()

    # Scale the image
    def wheelEvent(self, event):
        zoomFactor = 1.1 if event.angleDelta().y() > 0 else 0.9

        # Compute the new scale
        new_scale = self.transform().m11() * zoomFactor

        # Set a minimum scale limit (adjust as needed)
        min_scale = 0.1
        if new_scale < min_scale:
            zoomFactor = min_scale / self.transform().m11()

        self.scale(zoomFactor, zoomFactor)

        # Adjust the scene rectangle to ensure the image does not move out of view
        self.ensureVisible(self.sceneRect(), 0, 0)

    def mousePressEvent(self, event):
        # If scheme image is already selected
        if self.scheme_image is not None:
            # Move image
            if event.button() == Qt.LeftButton:
                self.setDragMode(QGraphicsView.ScrollHandDrag)
            # Select bounding box
            elif event.button() == Qt.RightButton:
                self.origin = self.mapToScene(event.pos()).toPoint()
                self.current_rubber_band = QRubberBand(QRubberBand.Rectangle, self)
                self.current_rubber_band.setGeometry(QRect(self.origin, QSize()))
                self.rubber_bands.append(self.current_rubber_band)

                # Set selection box color
                palette = QPalette()
                palette.setColor(QPalette.Highlight, Qt.green)
                self.current_rubber_band.setPalette(palette)

                self.current_rubber_band.show()
                self.draw_mode = True
        # Select an image file
        else:
            if event.button() == Qt.LeftButton:
                # Select image file
                file_dialog = QFileDialog()
                file_dialog.setNameFilter("Images (*.png *.xpm *.jpg *.bmp)")
                file_dialog.setViewMode(QFileDialog.Detail)

                # If file is selected
                if file_dialog.exec_():
                    file_path = file_dialog.selectedFiles()[0]
                    self.scheme_image = QPixmap(file_path)

                    pixmap_item = QGraphicsPixmapItem(self.scheme_image)
                    self.scene.addItem(pixmap_item)
                    self.setScene(self.scene)

        super().mousePressEvent(event)

    # Draw current selection box
    def mouseMoveEvent(self, event):
        if self.draw_mode:
            self.current_rubber_band.setGeometry(
                QRect(self.origin, self.mapToScene(event.pos()).toPoint()).normalized()
            )
        QGraphicsView.mouseMoveEvent(self, event)

    def mouseReleaseEvent(self, event):
        # Stop dragging the image
        if event.button() == Qt.LeftButton:
            self.setDragMode(QGraphicsView.NoDrag)
        # Finish drawing a selection box
        elif event.button() == Qt.RightButton:
            ic(self.current_rubber_band.geometry())
            self.draw_mode = False

        super().mouseReleaseEvent(event)

    def create_rect(self, start_point, end_point):
        x1 = min(start_point.x(), end_point.x())
        y1 = min(start_point.y(), end_point.y())
        x2 = max(start_point.x(), end_point.x())
        y2 = max(start_point.y(), end_point.y())

        rect = QPolygonF()
        rect.append(QPointF(x1, y1))
        rect.append(QPointF(x2, y1))
        rect.append(QPointF(x2, y2))
        rect.append(QPointF(x1, y2))

        return rect


if __name__ == "__main__":
    app = QApplication(sys.argv)
    scene = QGraphicsScene()
    view = SchemePreview(scene)
    view.show()
    sys.exit(app.exec_())


class SchemePreview(QGraphicsView):
    scheme_image: QPixmap
    rubber_bands = []
    current_rubber_band = None
    box_origin_point = None
    draw_mode = False

    def __init__(self, scene):
        super().__init__(scene)
        self.scene = QGraphicsScene(self)
        self.scheme_image = None

        # Zooming and moving the image
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setRenderHint(QPainter.TextAntialiasing)

        # Boxes selection
        self.setMouseTracking(True)
        self.rubber_bands = []
        self.current_rubber_band = None
        self.draw_mode = False
        self.origin = QPoint()

    # Scale the image
    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            zoomFactor = 1.1
        else:
            zoomFactor = 0.9

        self.scale(zoomFactor, zoomFactor)

    def mousePressEvent(self, event):
        # If scheme image is already selected
        if self.scheme_image is not None:

            # Move image
            if event.button() == Qt.LeftButton:
                self.setDragMode(QGraphicsView.ScrollHandDrag)

            # Select bounding box
            elif event.button() == Qt.RightButton:
                self.origin = event.pos()
                self.origin = QPoint(round(self.origin.x()), round(self.origin.y()))
                self.current_rubber_band = QRubberBand(QRubberBand.Rectangle, self)
                self.current_rubber_band.setGeometry(QRect(self.origin, QSize()))
                self.rubber_bands.append(self.current_rubber_band)

                palette = QPalette()
                palette.setColor(QPalette.Highlight, Qt.green)
                self.current_rubber_band.setPalette(palette)

                self.current_rubber_band.show()
                self.draw_mode = True

        # Select an image file
        else:
            if event.button() == Qt.LeftButton:
                # Select image file
                file_dialog = QFileDialog()
                file_dialog.setNameFilter("Images (*.png *.xpm *.jpg *.bmp)")
                file_dialog.setViewMode(QFileDialog.Detail)

                # If file is selected
                if file_dialog.exec_():
                    file_path = file_dialog.selectedFiles()[0]
                    self.scheme_image = QPixmap(file_path)

                    pixmap_item = QGraphicsPixmapItem(self.scheme_image)
                    self.scene.addItem(pixmap_item)
                    self.setScene(self.scene)

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.draw_mode:
            end_pos = event.pos()
            end_pos = QPoint(round(end_pos.x()), round(end_pos.y()))
            transformed_origin = self.mapToScene(self.origin)
            transformed_end_pos = self.mapToScene(end_pos)
            transformed_origin = QPoint(
                round(transformed_origin.x()), round(transformed_origin.y())
            )
            transformed_end_pos = QPoint(
                round(transformed_end_pos.x()), round(transformed_end_pos.y())
            )
            self.current_rubber_band.setGeometry(
                QRect(transformed_origin, transformed_end_pos).normalized()
            )
        QGraphicsView.mouseMoveEvent(self, event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.setDragMode(QGraphicsView.NoDrag)

        elif event.button() == Qt.RightButton:
            ic(self.current_rubber_band.geometry())
            self.draw_mode = False
            start_pos = self.mapToScene(self.current_rubber_band.geometry().topLeft())
            end_pos = self.mapToScene(self.current_rubber_band.geometry().bottomRight())
            self.drawRectangle(start_pos, end_pos)
            self.current_rubber_band.hide()

        super().mouseReleaseEvent(event)

    def drawRectangle(self, start_pos, end_pos):
        rect = self.create_rect(start_pos, end_pos)
        pen = QPen(Qt.red, 2)
        self.scene.addPolygon(rect, pen)

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

