from PyQt5.QtWidgets import (
    QWidget,
    QPushButton,
    QDesktopWidget,
    QProgressBar,
    QTableView,
    QMenu,
    QAction,
)
from PyQt5.QtGui import QFont


# Inherit this class and call self.set_styles() to automatically set color scheme, fonts and margins
# It is also possible to use custom colors, fonts and margins by calling corresponding methods
class StylizedWidget(QWidget):
    # ----------------------------------------------------------------
    # Color scheme
    bg_color: str = "#303030"
    fg_color: str = "#00A36C"

    # ----------------------------------------------------------------
    # Sets colors, fonts and margins for all UI elements
    def set_styles(self, bg_color=None, fg_color=None):
        self.apply_margins(self)
        self.set_colors(self)
        self.set_font(self)

        if bg_color is None:
            bg_color = self.bg_color

        if fg_color is None:
            fg_color = self.fg_color

        for widget in self.findChildren(QWidget):
            self.apply_margins(widget)
            self.set_colors(widget, bg_color, fg_color)
            self.set_font(widget)

    # Sets margins for specified element
    def apply_margins(self, widget, padx=7, pady=10):
        widget.setContentsMargins(padx, pady, padx, pady)

    # Sets colors for specified element
    def set_colors(self, widget, bg_color=None, fg_color=None):
        # If colors are not specified, use default
        if not bg_color:
            bg_color = self.bg_color

        if not fg_color:
            fg_color = self.fg_color

        if isinstance(widget, QProgressBar):
            # Remove glare animation and set progress bar color
            widget.setStyleSheet(
                f"""
                QProgressBar::chunk {{
                    background-color: {fg_color};
                    margin: 0px;
                }}
                """
            )

        elif isinstance(widget, QTableView):
            widget.setStyleSheet(
                f"""
                QTableWidget {{
                    background-color: {bg_color};
                    color: {fg_color};
                }}
                QHeaderView::section {{
                    background-color: {bg_color};
                    color: {fg_color};
                }}
                QTableWidget QTableCornerButton::section {{
                    background-color: {bg_color};
                }}
            """
            )
        
        elif isinstance(widget, QAction):
            widget.setStyleSheet(
                f"QPushButton {{ background-color: {bg_color}; color: {fg_color}; padding: 4px; }}"
            )
    

        else:
            widget.setAutoFillBackground(True)
            widget.setStyleSheet(f"background-color: {bg_color}; color: {fg_color};")

    # Sets fonts for specified element
    # family = "Comic Sans MS"
    def set_font(self, widget, family="Segoe UI", size=12):
        widget.setFont(QFont(family, size))

    # ----------------------------------------------------------------
    # Centers window on the screen
    def center(self):
        screen_geometry = QDesktopWidget().screenGeometry()
        center_x = (screen_geometry.width() - self.width()) // 2
        center_y = (screen_geometry.height() - self.height()) // 2
        self.move(center_x, center_y)
