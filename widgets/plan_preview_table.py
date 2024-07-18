from icecream import ic

from PyQt5.QtWidgets import (
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QMainWindow,
    QHBoxLayout,
    QLabel,
    QHeaderView,
    QPushButton,
    QMessageBox,
)
from PyQt5.QtGui import QPixmap
from PyQt5.Qt import Qt


class PlanPreviewTable(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Create the table widget
        main_layout = QVBoxLayout()
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Артикул", "Номенклатура", "Стоимость","Количество","Общая стоимость"])
        main_layout.addWidget(self.table)

        buttons_widget = QWidget(self)
        buttons_layout = QHBoxLayout(buttons_widget)
        buttons_widget.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(10)
        main_layout.addWidget(buttons_widget)

        # Create a button to add a new row
        self.add_row_button = QPushButton("Add a new element")
        self.add_row_button.clicked.connect(self.add_empty_line)
        buttons_layout.addWidget(self.add_row_button)

        # Create a button to delete selected rows
        self.delete_row_button = QPushButton("Delete selected")
        self.delete_row_button.clicked.connect(self.delete_selected_rows)
        buttons_layout.addWidget(self.delete_row_button)

        self.setLayout(main_layout)

        # Make the table resize with the window
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # self.table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def clear_table(self):
        """Clears all the content in the table."""
        self.table.clearContents()

    def add_row(self, row_data):
        """Adds a new row to the table.

        Args:
            row_data (tuple): A tuple containing 5 fields.

        """

        row_position = self.table.rowCount()
        self.table.insertRow(row_position)
        for col, data in enumerate(row_data):
            item = QTableWidgetItem(data)
            item.setTextAlignment(Qt.AlignCenter)  # Center the text
            self.table.setItem(row_position, col, item)

   
    def add_empty_line(self):
        """
        Adds a new line to the end of the table
        """

        self.add_row(["" for _ in range(5)])


    def update_table(self, article_occurences: dict[str, int], article_params: dict[str, tuple]):
        """
        Updates the table with the new article occurrences.
            article_occurences: {article: count}
            article_params: {article: (shkaf_name, nomenclature, price)}
        """

        self.clear_table()
        self.table.setRowCount(0)

        last_shkaf_name = None

        articles_and_shkaf_names = [
            (article, article_params[article][0]) for article in article_occurences.keys()
        ]
        sorted_articles = [item[0] for item in sorted(articles_and_shkaf_names, key=lambda x: x[0])]

        for article in sorted_articles:

            shkaf_name = article_params[article][0]

            if last_shkaf_name is None or shkaf_name != last_shkaf_name:
                self.add_row([shkaf_name, "", "", "", ""])
            
            quantity = article_occurences[article]

            nomenclature = article_params[article][1]
            cost = float(article_params[article][2])
            total_cost = cost * quantity

            self.add_row([article, nomenclature, str(cost), str(quantity), str(total_cost)])

            last_shkaf_name = shkaf_name

    def get_table_data(self):
        """
        Retrieves data from the table and returns it as a list of tuples.
        
        Returns:
            list: A list of tuples, where each tuple represents a row of data.
        """
        data = []
        for row in range(self.table.rowCount()):
            row_data = []
            for column in range(self.table.columnCount()):
                item = self.table.item(row, column)
                if item is not None:
                    row_data.append(item.text())
                else:
                    row_data.append("")
            data.append(tuple(row_data))
        return data

    def delete_selected_rows(self):
        """
        Deletes selected rows from the table
        """
        
        selected_rows = self.table.selectionModel().selectedRows()
        for row in sorted(selected_rows, reverse=True):
            self.table.removeRow(row.row())