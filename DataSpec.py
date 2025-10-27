import sys
import os
import random
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import mplcursors  


# Make sure you have run:
#    pyrcc5 resources.qrc -o resources_rc.py
import resources_rc

from PyQt5.QtWidgets import (
    QHBoxLayout,
    QCheckBox,
    QTabWidget,
    QApplication,
    QMainWindow,
    QFileDialog,
    QTableView,
    QAction,
    QMessageBox,
    QToolBar,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QStatusBar,
    QLineEdit,
    QDockWidget,
    QTextEdit,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QComboBox,
    QFormLayout,
    QLineEdit as QLEdit,
    QHeaderView,
    QMenu,
    QInputDialog,
    QColorDialog,
    QPushButton,
    QTabWidget,
    QSizePolicy,
    QSpinBox,
    QProgressDialog,
    QStackedWidget,
    QGraphicsOpacityEffect,
    QTextBrowser
)
from PyQt5.QtCore import (
    Qt,
    QModelIndex,
    pyqtSignal,
    QSize,
    QTimer,
    QPropertyAnimation,
    QSettings,
    QByteArray,
    QAbstractTableModel,
    QRectF
)
from PyQt5.QtGui import (
    QIcon,
    QBrush,
    QColor,
    QPainter,
    QFont,
    QFontMetrics,
    QPen
)

# Attempt to import the profiling library
try:
    from ydata_profiling import ProfileReport
except ImportError:
    ProfileReport = None

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from scipy.special import wofz

# Try to import PyMC/ArviZ for Bayesian linear regression
try:
    import pymc as pm
    import arviz as az
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False

# —– Fitting model definitions —–
def func_exp(x, a, b):
    return a * np.exp(b * x)

def func_gaussian(x, a, mu, sigma):
    return a * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def func_lorentzian(x, a, x0, gamma):
    return a * (gamma**2 / ((x - x0) ** 2 + gamma**2))

def func_voigt(x, a, mu, sigma, gamma):
    z = ((x - mu) + 1j * gamma) / (sigma * np.sqrt(2))
    return a * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))

# —————————————————————————————————
#  DARK & LIGHT QSS STYLES
# —————————————————————————————————

DARK_QSS = """
QWidget {
    background-color: #0D0D0D;
    color: #E0E0E0;
    font-family: 'Segoe UI', sans-serif;
    font-size: 14px;
}
QWidget#centralWidget {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #141414,
                                stop:1 #0D0D0D);
}
QToolBar {
    background: #141414;
    spacing: 4px;
    padding: 4px;
}
QToolBar QToolButton {
    color: #E0E0E0;
    background: #141414;
    border: none;
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 12px;
}
QToolBar QToolButton:hover {
    background: #1E1E1E;
}
QToolBar QToolButton:pressed {
    background: #008CBA;
    color: #0D0D0D;
}
QLineEdit {
    background-color: #141414;
    border: 1px solid #444444;
    color: #E0E0E0;
    border-radius: 4px;
    padding: 6px 8px;
    font-size: 14px;
}
QLineEdit:focus {
    border: 1px solid #008CBA;
}
QTableView {
    background-color: #0D0D0D;
    color: #E0E0E0;
    gridline-color: #2E2E2E;
    selection-background-color: #1E1E1E;
    border: none;
    selection-color: #E0E0E0;
}
QTableView::item:hover {
    background-color: #141414;
}
QHeaderView::section {
    background: #141414;
    color: #E0E0E0;
    padding: 4px;
    border: 1px solid #333333;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}
QDockWidget {
    background: #0D0D0D;
    font-family: 'Segoe UI', sans-serif;
    font-size: 14px;
    color: #E0E0E0;
    border: 1px solid #333333;
    border-radius: 4px;
}
QTextEdit {
    background: #141414;
    border: 1px solid #444444;
    color: #E0E0E0;
    border-radius: 4px;
    padding: 8px;
    font-size: 13px;
}
QPushButton {
    background: #141414;
    color: #E0E0E0;
    border: none;
    border-radius: 4px;
    padding: 6px 10px;
    font-size: 14px;
}
QPushButton:hover {
    background: #1E1E1E;
}
QDialog {
    background: #141414;
    border-radius: 8px;
}
QDialog QComboBox, QDialog QLineEdit {
    background: #0D0D0D;
    color: #E0E0E0;
    border: 1px solid #444444;
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 14px;
}
QDialog QPushButton {
    background: #141414;
    color: #E0E0E0;
    border: none;
    border-radius: 4px;
    padding: 6px 10px;
}
QDialog QPushButton:hover {
    background: #1E1E1E;
}
QTabWidget::pane {
    background: #0D0D0D;
    border: none;
}
QTabBar::tab {
    background: #141414;
    color: #E0E0E0;
    padding: 6px;
    border: 1px solid #333333;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    min-width: 80px;
}
QTabBar::tab:selected {
    background: #008CBA;
    color: #0D0D0D;
}
"""

LIGHT_QSS = """
QWidget {
    background-color: #FFFFFF;
    color: #000000;
    font-family: 'Segoe UI', sans-serif;
    font-size: 14px;
}
QWidget#centralWidget {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #FFFFFF,
                                stop:1 #F0F0F0);
}
QToolBar {
    background: #FFFFFF;
    spacing: 4px;
    padding: 4px;
}
QToolBar QToolButton {
    color: #000000;
    background: #FFFFFF;
    border: none;
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 12px;
}
QToolBar QToolButton:hover {
    background: #F0F0F0;
}
QToolBar QToolButton:pressed {
    background: #008CBA;
    color: #FFFFFF;
}
QLineEdit {
    background-color: #FFFFFF;
    border: 1px solid #CCCCCC;
    color: #000000;
    border-radius: 4px;
    padding: 6px 8px;
    font-size: 14px;
}
QLineEdit:focus {
    border: 1px solid #008CBA;
}
QTableView {
    background-color: #FFFFFF;
    color: #000000;
    gridline-color: #DDDDDD;
    selection-background-color: #CCCCCC;
    border: none;
    selection-color: #000000;
}
QTableView::item:hover {
    background-color: #F5F5F5;
}
QHeaderView::section {
    background: #F0F0F0;
    color: #000000;
    padding: 4px;
    border: 1px solid #CCCCCC;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}
QDockWidget {
    background: #FFFFFF;
    font-family: 'Segoe UI', sans-serif;
    font-size: 14px;
    color: #000000;
    border: 1px solid #CCCCCC;
    border-radius: 4px;
}
QTextEdit {
    background: #FFFFFF;
    border: 1px solid #CCCCCC;
    color: #000000;
    border-radius: 4px;
    padding: 8px;
    font-size: 13px;
}
QPushButton {
    background: #FFFFFF;
    color: #000000;
    border: none;
    border-radius: 4px;
    padding: 6px 10px;
    font-size: 14px;
}
QPushButton:hover {
    background: #F0F0F0;
}
QDialog {
    background: #FFFFFF;
    border-radius: 8px;
}
QDialog QComboBox, QDialog QLineEdit {
    background: #FFFFFF;
    color: #000000;
    border: 1px solid #CCCCCC;
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 14px;
}
QDialog QPushButton {
    background: #FFFFFF;
    color: #000000;
    border: none;
    border-radius: 4px;
    padding: 6px 10px;
}
QDialog QPushButton:hover {
    background: #F0F0F0;
}
QTabWidget::pane {
    background: #FFFFFF;
    border: none;
}
QTabBar::tab {
    background: #F0F0F0;
    color: #000000;
    padding: 6px;
    border: 1px solid #CCCCCC;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    min-width: 80px;
}
QTabBar::tab:selected {
    background: #008CBA;
    color: #FFFFFF;
}
"""


# —————————————————————————————————
#  PROGRAMMATIC LOGO WIDGET
# —————————————————————————————————

from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui     import QPainter, QColor, QFont, QFontMetrics, QPen
from PyQt5.QtCore    import QRectF, Qt

# —————————————————————————————————————————————————————————————————————
#  LOGO WIDGET (Bars + Blue Line + Cursive “DATASPEC” + Eurostile tagline)
# —————————————————————————————————————————————————————————————————————

class LogoWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)

        # ─── FONTS ───────────────────────────────────────────────────────────
        # Title: Cursive “Brush Script MT” (fallback: Verdana)
        # Tagline: Eurostile (modern, minimalistic; fallback: Helvetica Neue)
        self.title_font_family   = "Verdana"
        self.tagline_font_family = "Helvetica Neue"
        # ─────────────────────────────────────────────────────────────────────


    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 1) Determine dark vs. light mode
        is_dark = False
        mw = self.window()
        if hasattr(mw, "is_dark_mode"):
            is_dark = mw.is_dark_mode

        if is_dark:
            bar_color       = QColor("#FFFFFF")
            line_color      = QColor("#008CBA")
            endpoint_color  = QColor("#008CBA")
            text_color      = QColor("#FFFFFF")
            underline_color = QColor("#FFFFFF")
        else:
            bar_color       = QColor("#000000")
            line_color      = QColor("#008CBA")
            endpoint_color  = QColor("#008CBA")
            text_color      = QColor("#000000")
            underline_color = QColor("#000000")

        w = self.width()
        h = self.height()

        # ——— 2) Build an aggressive (scale=0.60) mini‐logo ≈18% of height ———

        Wv = w * 0.80    # virtual width (80% of widget)
        Hv = h * 0.30    # virtual height (30% of widget)
        scale = 0.60     # enlarge from 0.55 → 0.60

        Ws = Wv * scale  # on-screen mini‐logo width
        Hs = Hv * scale  # on-screen mini‐logo height (≈0.18h)

        offset_x = (w - Ws) / 2.1  # center horizontally
        offset_y = h * 0.05      # 5% margin from top

        # ——— 2a) Bar widths & spacing (25% thinner, then 10%/90% split) ———
        total_bar_width = Ws * 0.35
        narrow_group    = total_bar_width * 0.75
        bar_w           = narrow_group / 3.0
        rem_space       = total_bar_width - narrow_group

        spacing1 = rem_space * 0.2   # 10% between Bar1⇢Bar2
        spacing2 = rem_space * 1.75  # 90% between Bar2⇢Bar3

        # ——— 2b) Bar heights (3× Hv) ———
        bar_max_height = Hv * 3.0
        heights = [
            0.25 * bar_max_height,  # 25%
            0.45 * bar_max_height,  # 45%
            0.75 * bar_max_height   # 65%
        ]

        # ——— 2c) Vertical offset within mini‐logo (10% of Hv) ———
        local_y_offset = (Hv * 0.10)

        # ——— 2d) Build virtual QRectF for each bar, then project onto screen ———
        start_x_v = (Wv - total_bar_width) / 2.0

        r1_v = QRectF(
            start_x_v,
            local_y_offset + (bar_max_height - heights[0]),
            bar_w,
            heights[0]
        )
        r2_v = QRectF(
            start_x_v + bar_w + spacing1,
            local_y_offset + (bar_max_height - heights[1]),
            bar_w,
            heights[1]
        )
        r3_v = QRectF(
            start_x_v + 2*bar_w + spacing1 + spacing2,
            local_y_offset + (bar_max_height - heights[2]),
            bar_w,
            heights[2]
        )

        def to_screen_rect(rv: QRectF) -> QRectF:
            return QRectF(
                offset_x + rv.x() * scale,
                offset_y + rv.y() * scale,
                rv.width() * scale,
                rv.height() * scale
            )

        rect1 = to_screen_rect(r1_v)
        rect2 = to_screen_rect(r2_v)
        rect3 = to_screen_rect(r3_v)

        # ——— 2e) Draw the three bars ———
        painter.setBrush(bar_color)
        painter.setPen(Qt.NoPen)
        for r in (rect1, rect2, rect3):
            painter.drawRect(r)

        # ——— 2f) Draw polyline that “rises, crashes down, then rises,” floating high ———
        # Increase line_offset = Hs * 0.5 so the blue line never touches the bars
        line_offset = Hs * 0.5

        cx1 = rect1.x() + rect1.width() / 2.5
        cy1 = rect1.y() - line_offset
        cx2 = rect2.x() + rect2.width() / 2.5
        cy2 = rect2.y() - line_offset
        cx3 = rect3.x() + rect3.width() / 3.5
        cy3 = rect3.y() - line_offset

        # Crash‐point between Bar2⇢Bar3, 40% of Hs below cy2
        crash_x = (cx2 + cx3) / 2.02
        crash_y = cy2 + (Hs * 0.40)

        painter.setPen(QPen(line_color, 4, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.drawLine(int(cx1), int(cy1), int(cx2), int(cy2))         # Bar1→Bar2
        painter.drawLine(int(cx2), int(cy2), int(crash_x), int(crash_y)) # Bar2→Crash
        painter.drawLine(int(crash_x), int(crash_y), int(cx3), int(cy3)) # Crash→Bar3

        # Draw endpoint circles at Bar1 & Bar3 only
        dot_radius = 20.0
        painter.setBrush(endpoint_color)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QRectF(cx1 - dot_radius/2.0, cy1 - dot_radius/2.0, dot_radius, dot_radius))
        painter.drawEllipse(QRectF(cx3 - dot_radius/2.0, cy3 - dot_radius/2.0, dot_radius, dot_radius))

        # ——— 3) Draw “DATASPEC” in cursive (Verdana fallback), 18.75% of height ———
        title_font_pt = max(14, int(h * 0.1875))  # ~18.75% of h
        title_font = QFont(self.title_font_family, title_font_pt)
        painter.setFont(title_font)
        painter.setPen(text_color)

        title_str = "DATASPEC"
        fm_title = QFontMetrics(title_font)
        title_w = fm_title.horizontalAdvance(title_str)
        title_ascent = fm_title.ascent()

        # Push “DATASPEC” down by 42% of widget height under the mini‐logo
        baseline_y = offset_y + Hs + (h * 0.42) + title_ascent
        title_x = (w - title_w) / 2.0
        painter.drawText(int(title_x), int(baseline_y), title_str)

        # ——— 4) Draw catchphrase (Eurostile → fallback Helvetica Neue), ~9.375% of height ———
        tagline_font_pt = max(10, int(h * 0.09375))  # ~9.375% of h
        tagline_font = QFont(self.tagline_font_family, tagline_font_pt)
        painter.setFont(tagline_font)
        painter.setPen(text_color)

        tagline_str = "WHERE INSIGHT MEETS PRECISION"
        fm_tag = QFontMetrics(tagline_font)
        tag_w = fm_tag.horizontalAdvance(tagline_str)
        tag_ascent = fm_tag.ascent()

        # Gap of 3% of h under “DATASPEC”
        tagline_baseline_y = baseline_y + tag_ascent + (h * 0.03)
        tagline_x = (w - tag_w) / 2.0
        painter.drawText(int(tagline_x), int(tagline_baseline_y), tagline_str)

        # ——— 5) Draw a very short underline (20% width) under the catchphrase ———
        ul_margin = w * 0.40  # central 20%
        ul_y = tagline_baseline_y + 4
        painter.setPen(QPen(underline_color, 2))
        painter.drawLine(int(ul_margin), int(ul_y), int(w - ul_margin), int(ul_y))

        painter.end()


# —————————————————————————————————
#  PANDAS MODEL
# —————————————————————————————————

class PandasModel(QAbstractTableModel):
    data_changed = pyqtSignal()
    cell_edited = pyqtSignal(int, int, object)

    def __init__(self, df=pd.DataFrame(), workflow_callback=None, parent=None):
        super().__init__(parent)
        self._df = df.copy()
        self._original_df = df.copy()
        self.workflow_callback = workflow_callback
        self.conditional_rules = []

    def update_dataframe(self, new_df):
        self.beginResetModel()
        self._df = new_df.copy()
        self._original_df = new_df.copy()
        self.endResetModel()
        self.data_changed.emit()

    def rowCount(self, parent=QModelIndex()):
        return len(self._df)

    def columnCount(self, parent=QModelIndex()):
        return len(self._df.columns)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        value = self._df.iat[index.row(), index.column()]

        if role == Qt.DisplayRole:
            return "" if pd.isna(value) else str(value)

        if role == Qt.BackgroundRole:
            for col, op, val, color in self.conditional_rules:
                if self._df.columns[index.column()] == col:
                    cell_value = value
                    try:
                        cell_num = float(cell_value)
                        val_num = float(val)
                        cond = False
                        if op == ">":
                            cond = cell_num > val_num
                        elif op == "<":
                            cond = cell_num < val_num
                        elif op == ">=":
                            cond = cell_num >= val_num
                        elif op == "<=":
                            cond = cell_num <= val_num
                        elif op == "==":
                            cond = cell_num == val_num
                        elif op == "!=":
                            cond = cell_num != val_num
                        if cond:
                            return QBrush(QColor(color))
                    except Exception:
                        if op == "contains" and pd.notna(cell_value):
                            if val.lower() in str(cell_value).lower():
                                return QBrush(QColor(color))
            return None

        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return str(self._df.columns[section])
        else:
            return str(self._df.index[section])

    def sort(self, column, order):
        if self._df.shape[1] == 0 or not (0 <= column < self._df.shape[1]):
            return
        col_name = self._df.columns[column]
        ascending = (order == Qt.AscendingOrder)
        self.layoutAboutToBeChanged.emit()
        self._df = (
            self._df.sort_values(by=col_name, ascending=ascending)
            .reset_index(drop=True)
        )
        self._original_df = self._df.copy()
        self.layoutChanged.emit()
        self.data_changed.emit()

    def flags(self, index):
        if not index.isValid():
            return Qt.ItemIsEnabled
        return Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable

    def setData(self, index, value, role=Qt.EditRole):
        if index.isValid() and role == Qt.EditRole:
            row, col = index.row(), index.column()
            col_name = self._df.columns[col]
            try:
                dtype = self._df[col_name].dtype
                if pd.api.types.is_numeric_dtype(dtype):
                    new_val = float(value)
                    if dtype == int:
                        new_val = int(new_val)
                else:
                    new_val = value
            except Exception:
                new_val = value

            self._df.iat[row, col] = new_val
            self._original_df.iat[row, col] = new_val

            self.data_changed.emit()
            self.cell_edited.emit(row, col, new_val)
            self.dataChanged.emit(index, index, [Qt.DisplayRole])
            return True
        return False

    def removeRows(self, row_indices):
        for row in sorted(row_indices, reverse=True):
            self.beginRemoveRows(QModelIndex(), row, row)
            self._df.drop(self._df.index[row], inplace=True)
            self.endRemoveRows()
        self._df.reset_index(drop=True, inplace=True)
        self._original_df = self._df.copy()
        self.data_changed.emit()

    def dropAllNARows(self):
        self.beginResetModel()
        self._df = self._df.dropna().reset_index(drop=True)
        self._original_df = self._df.copy()
        self.endResetModel()
        self.data_changed.emit()

    def fillNARows(self, method, constant=None):
        self.beginResetModel()
        if method == "Mean":
            self._df = self._df.fillna(self._df.mean(numeric_only=True))
        elif method == "Median":
            self._df = self._df.fillna(self._df.median(numeric_only=True))
        elif method == "Forward Fill":
            self._df = self._df.fillna(method="ffill")
        elif method == "Backward Fill":
            self._df = self._df.fillna(method="bfill")
        elif method == "Constant":
            self._df = self._df.fillna(constant)

        self._original_df = self._df.copy()
        self.endResetModel()
        self.data_changed.emit()

    def renameColumn(self, old_name, new_name):
        self.beginResetModel()
        self._df.rename(columns={old_name: new_name}, inplace=True)
        self._original_df = self._df.copy()
        self.endResetModel()
        self.data_changed.emit()

    def deleteColumn(self, col_index):
        self.beginResetModel()
        col_name = self._df.columns[col_index]
        self._df.drop(columns=[col_name], inplace=True)
        self._original_df = self._df.copy()
        self.endResetModel()
        self.data_changed.emit()

    def filter(self, text):
        self.beginResetModel()
        if text == "":
            self._df = self._original_df.copy()
        else:
            mask = self._original_df.apply(
                lambda row: row.astype(str).str.contains(text, case=False, na=False).any(),
                axis=1,
            )
            self._df = self._original_df[mask].reset_index(drop=True)
        self.endResetModel()
        self.data_changed.emit()

    def addConditionalRule(self, column, operator, value, color):
        self.conditional_rules.append((column, operator, value, color))
        self.data_changed.emit()

    def getDataFrame(self):
        return self._df.copy()

    def getOriginalDataFrame(self):
        return self._original_df.copy()


# ─────────────────────────────────────────────────────────
# SettingsDialog: combined “Settings” + keyboard shortcuts + help
# ─────────────────────────────────────────────────────────
class SettingsDialog(QDialog):
    def __init__(self, parent=None, shortcuts=None):
        super().__init__(parent)
        self.setWindowTitle("Settings & Shortcuts")
        self.resize(600, 400)

        tabs = QTabWidget(self)

        # --- Tab 1: Shortcuts ---
        tab1 = QWidget()
        lay1 = QVBoxLayout(tab1)
        lay1.addWidget(QLabel("<h2>Keyboard Shortcuts</h2>"))
        for name, key in (shortcuts or {}).items():
            lay1.addWidget(QLabel(f"<b>{name}:</b> {key}"))
        tabs.addTab(tab1, "Shortcuts")

        # --- Tab 2: Overview / Help ---
        tab2 = QWidget()
        lay2 = QVBoxLayout(tab2)
        help_html = (
            "<h2>What This Software Can Do</h2>"
            "<ul>"
            "<li>Drag &amp; drop CSV, TXT, DAT files</li>"
            "<li>Smart context menus &amp; right-click actions</li>"
            "<li>Interactive advanced plotting:</li>"
            "  <ul>"
            "    <li>Linear / Quadratic / Cubic ﬁts</li>"
            "    <li>Gaussian &amp; Exponential (non-linear) ﬁts</li>"
            "    <li>Theme toggle, color pickers, hover tooltips</li>"
            "    <li>Zoom / pan &amp; Download button</li>"
            "  </ul>"
            "<li>Dashboard with heatmaps, box &amp; scatter plots</li>"
            "<li>Settings to customize shortcuts &amp; behavior</li>"
            "</ul>"
        )
        lay2.addWidget(QLabel(help_html))
        tabs.addTab(tab2, "Help")

        # Close button
        btns = QDialogButtonBox(QDialogButtonBox.Close)
        btns.rejected.connect(self.close)

        # Main layout
        main_lay = QVBoxLayout(self)
        main_lay.addWidget(tabs)
        main_lay.addWidget(btns)


# —————————————————————————————————
#  FILL NA DIALOG
# —————————————————————————————————

class FillNADialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Fill Missing Values")
        self.setFixedWidth(320)
        self.setStyleSheet("""
            QDialog {
                background: #141414;
                border-radius: 8px;
            }
            QComboBox, QLineEdit, QPushButton {
                background: #0D0D0D;
                color: #E0E0E0;
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 14px;
            }
            QPushButton:hover {
                background: #1E1E1E;
            }
        """)
        layout = QFormLayout(self)
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Mean", "Median", "Forward Fill", "Backward Fill", "Constant"])
        layout.addRow("Method:", self.method_combo)
        self.constant_input = QLEdit()
        self.constant_input.setPlaceholderText("Only if Constant")
        layout.addRow("Constant:", self.constant_input)
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def getValues(self):
        return self.method_combo.currentText(), self.constant_input.text()


# —————————————————————————————————
#  CONDITIONAL FORMATTING DIALOG
# —————————————————————————————————

class ConditionalDialog(QDialog):
    def __init__(self, columns, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Conditional Formatting Rule")
        self.setFixedWidth(360)
        self.setStyleSheet("""
            QDialog {
                background: #141414;
                border-radius: 8px;
            }
            QComboBox, QLineEdit, QPushButton {
                background: #0D0D0D;
                color: #E0E0E0;
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 14px;
            }
            QPushButton:hover {
                background: #1E1E1E;
            }
        """)
        layout = QFormLayout(self)
        self.column_combo = QComboBox()
        self.column_combo.addItems(columns)
        layout.addRow("Column:", self.column_combo)
        self.operator_combo = QComboBox()
        self.operator_combo.addItems([">", "<", ">=", "<=", "==", "!=", "contains"])
        layout.addRow("Operator:", self.operator_combo)
        self.value_input = QLEdit()
        layout.addRow("Value:", self.value_input)
        self.color_button = QPushButton("Select Color")
        self.selected_color = "#008CBA"
        self.color_button.setStyleSheet(f"background: {self.selected_color}; color: #0D0D0D;")
        self.color_button.clicked.connect(self.chooseColor)
        layout.addRow("Color:", self.color_button)
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def chooseColor(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.selected_color = color.name()
            self.color_button.setStyleSheet(f"background: {self.selected_color}; color: #0D0D0D;")

    def getValues(self):
        return (
            self.column_combo.currentText(),
            self.operator_combo.currentText(),
            self.value_input.text(),
            self.selected_color,
        )


# —————————————————————————————————
#  VISUALIZATION & DASHBOARD DIALOGS
# —————————————————————————————————

class VisualizeDialog(QDialog):
    def __init__(self, df, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Visualize Column")
        self.df = df
        self.setMinimumSize(600, 480)
        self.setStyleSheet("""
            QDialog {
                background: #141414;
                border-radius: 8px;
            }
            QComboBox, QPushButton {
                background: #0D0D0D;
                color: #E0E0E0;
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 14px;
            }
            QPushButton:hover {
                background: #1E1E1E;
            }
        """)
        layout = QVBoxLayout(self)
        form_layout = QFormLayout()
        self.column_combo = QComboBox()
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        self.column_combo.addItems(numeric_cols)
        form_layout.addRow("Numeric Column:", self.column_combo)
        layout.addLayout(form_layout)
        self.canvas = FigureCanvas(Figure(figsize=(5, 4)))
        layout.addWidget(self.canvas)
        self.ax = self.canvas.figure.subplots()
        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        button_box.rejected.connect(self.close)
        layout.addWidget(button_box)
        self.column_combo.currentTextChanged.connect(self.plot_histogram)
        if numeric_cols:
            self.plot_histogram(numeric_cols[0])

    def plot_histogram(self, col_name):
        self.ax.clear()
        series = self.df[col_name].dropna()
        self.ax.hist(series, bins=20, edgecolor="#333333", color="#008CBA")
        self.ax.set_title(f"Histogram: {col_name}", color="#E0E0E0")
        self.ax.set_xlabel(col_name, color="#E0E0E0")
        self.ax.set_ylabel("Frequency", color="#E0E0E0")
        self.ax.tick_params(colors="#E0E0E0")
        self.canvas.draw()


class DashboardDialog(QDialog):
    def __init__(self, df, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Dashboard")
        self.df = df
        self.setMinimumSize(880, 680)
        self.setStyleSheet("""
            QDialog {
                background: #141414;
                border-radius: 8px;
            }
            QComboBox, QPushButton {
                background: #0D0D0D;
                color: #E0E0E0;
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 14px;
            }
            QPushButton:hover {
                background: #1E1E1E;
            }
        """)
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane { background: #0D0D0D; border: none; }
            QTabBar::tab {
                background: #141414; color: #E0E0E0; padding: 6px;
                border: 1px solid #333333; border-bottom: none;
                border-top-left-radius: 4px; border-top-right-radius: 4px;
                min-width: 80px;
            }
            QTabBar::tab:selected {
                background: #008CBA; color: #0D0D0D;
            }
        """)
        layout.addWidget(tabs)

        # 1) Correlation Heatmap
        corr_tab = QWidget()
        corr_layout = QVBoxLayout(corr_tab)
        self.corr_canvas = FigureCanvas(Figure(figsize=(5, 4)))
        corr_layout.addWidget(self.corr_canvas)
        self.plot_correlation()
        tabs.addTab(corr_tab, "Correlation")

        # 2) Box Plot
        box_tab = QWidget()
        box_layout = QVBoxLayout(box_tab)
        form_layout = QFormLayout()
        self.box_combo = QComboBox()
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        self.box_combo.addItems(numeric_cols)
        form_layout.addRow("Numeric Column:", self.box_combo)
        box_layout.addLayout(form_layout)
        self.box_canvas = FigureCanvas(Figure(figsize=(5, 4)))
        box_layout.addWidget(self.box_canvas)
        self.box_combo.currentTextChanged.connect(self.plot_box)
        if numeric_cols:
            self.plot_box(numeric_cols[0])
        tabs.addTab(box_tab, "Box Plot")

        # 3) Scatter Plot
        scatter_tab = QWidget()
        scatter_layout = QVBoxLayout(scatter_tab)
        form_layout2 = QFormLayout()
        self.scatter_x = QComboBox()
        self.scatter_y = QComboBox()
        self.scatter_x.addItems(numeric_cols)
        self.scatter_y.addItems(numeric_cols)
        form_layout2.addRow("X-axis:", self.scatter_x)
        form_layout2.addRow("Y-axis:", self.scatter_y)
        scatter_layout.addLayout(form_layout2)
        self.scatter_canvas = FigureCanvas(Figure(figsize=(5, 4)))
        scatter_layout.addWidget(self.scatter_canvas)
        self.scatter_x.currentTextChanged.connect(self.plot_scatter)
        self.scatter_y.currentTextChanged.connect(self.plot_scatter)
        if len(numeric_cols) >= 2:
            self.plot_scatter()
        tabs.addTab(scatter_tab, "Scatter")

    def plot_correlation(self):
        self.corr_canvas.figure.clear()
        ax = self.corr_canvas.figure.subplots()
        corr = self.df.select_dtypes(include="number").corr()
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right", color="#E0E0E0")
        ax.set_yticks(range(len(corr.index)))
        ax.set_yticklabels(corr.index, color="#E0E0E0")
        cbar = self.corr_canvas.figure.colorbar(im, ax=ax)
        cbar.ax.yaxis.set_tick_params(color="#E0E0E0")
        for tick in cbar.ax.get_yticklabels():
            tick.set_color("#E0E0E0")
        ax.set_title("Correlation Heatmap", color="#E0E0E0")
        self.corr_canvas.draw()

    def plot_box(self, col_name):
        self.box_canvas.figure.clear()
        ax = self.box_canvas.figure.subplots()
        series = self.df[col_name].dropna()
        ax.boxplot(
            series,
            vert=True,
            patch_artist=True,
            boxprops=dict(facecolor="#008CBA", edgecolor="#E0E0E0"),
            medianprops=dict(color="#333333"),
        )
        ax.set_title(f"Box Plot: {col_name}", color="#E0E0E0")
        ax.set_ylabel(col_name, color="#E0E0E0")
        ax.tick_params(colors="#E0E0E0")
        self.box_canvas.draw()

    def plot_scatter(self):
        x_col = self.scatter_x.currentText()
        y_col = self.scatter_y.currentText()
        if x_col and y_col:
            self.scatter_canvas.figure.clear()
            ax = self.scatter_canvas.figure.subplots()
            ax.scatter(self.df[x_col], self.df[y_col], alpha=0.8, color="#008CBA", edgecolor="#E0E0E0")
            ax.set_xlabel(x_col, color="#E0E0E0")
            ax.set_ylabel(y_col, color="#E0E0E0")
            ax.set_title(f"Scatter: {x_col} vs {y_col}", color="#E0E0E0")
            ax.tick_params(colors="#E0E0E0")
            self.scatter_canvas.draw()



class ExploreDialog(QDialog):
    def __init__(self, df, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Explore Data")
        self.df = df
        self.setMinimumSize(1000, 750)

        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        layout.addWidget(tabs)

        # 1) Histogram
        self.hist_canvas = FigureCanvas(Figure(figsize=(6, 4)))
        self._add_tab(tabs, "Histogram", self.hist_canvas, self.plot_histogram)

        # 2) Box Plot
        self.box_canvas = FigureCanvas(Figure(figsize=(6, 4)))
        self._add_tab(tabs, "Box Plot", self.box_canvas, self.plot_box)

        # 3) Scatter
        self.scatter_canvas = FigureCanvas(Figure(figsize=(6, 4)))
        self._add_tab(tabs, "Scatter", self.scatter_canvas, self.plot_scatter)

        # 4) Correlation Heatmap
        self.corr_canvas = FigureCanvas(Figure(figsize=(6, 4)))
        self._add_tab(tabs, "Correlation", self.corr_canvas, self.plot_correlation)

        # 5) Pair Plot (all numeric variables)
        self.pair_canvas = FigureCanvas(Figure(figsize=(6, 4)))
        self._add_tab(tabs, "Pair Plot", self.pair_canvas, self.plot_pairplot)

        # 6) Violin Plot
        self.violin_canvas = FigureCanvas(Figure(figsize=(6, 4)))
        self._add_tab(tabs, "Violin Plot", self.violin_canvas, self.plot_violin)

        # 7) Time Series (if datetime index or columns)
        self.ts_canvas = FigureCanvas(Figure(figsize=(6, 4)))
        self._add_tab(tabs, "Time Series", self.ts_canvas, self.plot_timeseries)

    def _add_tab(self, tabs, name, canvas, plot_func):
        tab = QWidget()
        vbox = QVBoxLayout(tab)
        vbox.addWidget(canvas)
        tabs.addTab(tab, name)
        # Plot immediately
        plot_func()

    def plot_histogram(self):
        self.hist_canvas.figure.clear()
        ax = self.hist_canvas.figure.subplots()
        num_cols = self.df.select_dtypes(include="number").columns
        if len(num_cols) > 0:
            self.df[num_cols[0]].hist(ax=ax, bins=30, color="#008CBA", edgecolor="black")
            ax.set_title(f"Histogram: {num_cols[0]}")
        self.hist_canvas.draw()

    def plot_box(self):
        self.box_canvas.figure.clear()
        ax = self.box_canvas.figure.subplots()
        num_cols = self.df.select_dtypes(include="number").columns
        if len(num_cols) > 0:
            self.df[num_cols].plot(kind="box", ax=ax, patch_artist=True, color={"boxes":"#008CBA"})
            ax.set_title("Box Plots")
        self.box_canvas.draw()

    def plot_scatter(self):
        self.scatter_canvas.figure.clear()
        ax = self.scatter_canvas.figure.subplots()
        num_cols = self.df.select_dtypes(include="number").columns
        if len(num_cols) >= 2:
            ax.scatter(self.df[num_cols[0]], self.df[num_cols[1]], color="#008CBA", alpha=0.7)
            ax.set_xlabel(num_cols[0]); ax.set_ylabel(num_cols[1])
            ax.set_title("Scatter Plot")
        self.scatter_canvas.draw()

    def plot_correlation(self):
        self.corr_canvas.figure.clear()
        ax = self.corr_canvas.figure.subplots()
        corr = self.df.select_dtypes(include="number").corr()
        cax = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(corr.index)))
        ax.set_yticklabels(corr.index)
        self.corr_canvas.figure.colorbar(cax, ax=ax)
        ax.set_title("Correlation Heatmap")
        self.corr_canvas.draw()

    def plot_pairplot(self):
        self.pair_canvas.figure.clear()
        ax = self.pair_canvas.figure.subplots()
        num_cols = self.df.select_dtypes(include="number").columns
        if len(num_cols) > 1:
            pd.plotting.scatter_matrix(self.df[num_cols], ax=ax, alpha=0.7, diagonal="hist")
            ax.set_title("Pair Plot")
        self.pair_canvas.draw()

    def plot_violin(self):
        self.violin_canvas.figure.clear()
        ax = self.violin_canvas.figure.subplots()
        num_cols = self.df.select_dtypes(include="number").columns
        if len(num_cols) > 0:
            ax.violinplot([self.df[col].dropna() for col in num_cols], showmeans=True)
            ax.set_xticks(range(1, len(num_cols) + 1))
            ax.set_xticklabels(num_cols, rotation=45)
            ax.set_title("Violin Plots")
        self.violin_canvas.draw()

    def plot_timeseries(self):
        self.ts_canvas.figure.clear()
        ax = self.ts_canvas.figure.subplots()
        date_cols = self.df.select_dtypes(include="datetime").columns
        num_cols = self.df.select_dtypes(include="number").columns
        if len(date_cols) > 0 and len(num_cols) > 0:
            self.df.plot(x=date_cols[0], y=num_cols[0], ax=ax, color="#008CBA")
            ax.set_title("Time Series")
        self.ts_canvas.draw()


# —————————————————————————————————
#  Fit DIALOGS
# —————————————————————————————————

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QDialog, QDockWidget,
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QLineEdit, QTextEdit, QCheckBox,
    QFileDialog, QMessageBox, QColorDialog,
    QListWidget, QListWidgetItem   # <-- add these
)

# ──────────────────────────────────────────────────────────────────────────────
# NEW FIT PANEL (no modal dialog, no black window)
# ──────────────────────────────────────────────────────────────────────────────
from dataclasses import dataclass
from typing import Tuple, Dict, Callable
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import wofz

from PyQt5.QtWidgets import (
    QWidget, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QDoubleSpinBox, QCheckBox, QPushButton, QPlainTextEdit, QFileDialog,
    QGroupBox, QFormLayout, QSplitter
)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector


# ---------- model functions ----------
def _exp(x, a, b):              # y = a * exp(b x)
    return a * np.exp(b * x)

def _gauss(x, a, mu, sigma):    # a * exp(-(x-mu)^2 / (2 sigma^2))
    return a * np.exp(-((x - mu)**2) / (2.0 * sigma**2))

def _lorentz(x, a, x0, gamma):  # a * gamma^2 / ((x-x0)^2 + gamma^2)
    return a * (gamma**2 / ((x - x0)**2 + gamma**2))

def _voigt(x, a, mu, sigma, gamma):
    z = ((x - mu) + 1j * gamma) / (sigma * np.sqrt(2))
    return a * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))


# ---------- helpers ----------
@dataclass
class FitResult:
    name: str
    params: Dict[str, float]
    x: np.ndarray
    y_fit: np.ndarray
    cov: np.ndarray | None
    r2: float | None


def _r2_score(y, yhat) -> float:
    ss_res = np.sum((y - yhat) ** 2.0)
    ss_tot = np.sum((y - np.mean(y)) ** 2.0)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan


# ──────────────────────────────────────────────────────────────────────────────
# FitDialog: lightweight QWidget (no modal QDialog) + immediate plotting
# ──────────────────────────────────────────────────────────────────────────────
class FitDialog(QWidget):
    """
    Fresh, minimal, reliable fit panel.
    - Immediate scatter on open
    - Drag a range to fit (SpanSelector) or use spinboxes
    - Overlays chosen fit + optional 95% CI
    - PNG export
    """
    def __init__(self, df: pd.DataFrame, parent=None):
        super().__init__(parent)
        self.df = df.copy()

        # Layout scaffold (splitter: plot left | results right)
        root = QVBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # Left side: controls + canvas
        left = QWidget(); left_lay = QVBoxLayout(left); left_lay.setContentsMargins(0,0,0,0)
        splitter.addWidget(left)

        # Right side: results
        self.results = QPlainTextEdit(); self.results.setReadOnly(True)
        splitter.addWidget(self.results)
        splitter.setSizes([800, 300])

        # Controls
        ctrl = QGroupBox("Controls")
        form = QFormLayout(ctrl)
        left_lay.addWidget(ctrl)

        numeric = self.df.select_dtypes(include="number").columns.tolist()
        if len(numeric) < 2:
            numeric = ["", ""]  # guard, but window will still show

        self.x_combo = QComboBox(); self.x_combo.addItems(numeric)
        self.y_combo = QComboBox(); self.y_combo.addItems(numeric)
        form.addRow(QLabel("X column:"), self.x_combo)
        form.addRow(QLabel("Y column:"), self.y_combo)

        # X interval selectors
        self.xmin_spin = QDoubleSpinBox(); self.xmin_spin.setDecimals(3)
        self.xmax_spin = QDoubleSpinBox(); self.xmax_spin.setDecimals(3)

        # Y interval selectors
        self.ymin_spin = QDoubleSpinBox(); self.ymin_spin.setDecimals(3)
        self.ymax_spin = QDoubleSpinBox(); self.ymax_spin.setDecimals(3)

        # Set ranges based on data
        xmin_global, xmax_global = float(df[numeric].min().min()), float(df[numeric].max().max())
        ymin_global, ymax_global = float(df[numeric].min().min()), float(df[numeric].max().max())

        self.xmin_spin.setRange(xmin_global, xmax_global)
        self.xmax_spin.setRange(xmin_global, xmax_global)
        self.ymin_spin.setRange(ymin_global, ymax_global)
        self.ymax_spin.setRange(ymin_global, ymax_global)

        self.xmin_spin.setValue(xmin_global)
        self.xmax_spin.setValue(xmax_global)
        self.ymin_spin.setValue(ymin_global)
        self.ymax_spin.setValue(ymax_global)

        # Layouts
        row_interval = QHBoxLayout()
        row_interval.addWidget(QLabel("X range:"))
        row_interval.addWidget(self.xmin_spin)
        row_interval.addWidget(self.xmax_spin)
        form.addRow(row_interval)

        row_interval_y = QHBoxLayout()
        row_interval_y.addWidget(QLabel("Y range:"))
        row_interval_y.addWidget(self.ymin_spin)
        row_interval_y.addWidget(self.ymax_spin)
        form.addRow(row_interval_y)

        # Model & options
        opt_row1 = QHBoxLayout()
        self.model = QComboBox(); self.model.addItems(
            ["Linear", "Polynomial (deg 2)", "Polynomial (deg 3)",
             "Exponential", "Gaussian", "Lorentzian", "Voigt", "Custom"]
        )
        self.show_ci = QCheckBox("95% CI")
        opt_row1.addWidget(QLabel("Model:")); opt_row1.addWidget(self.model); opt_row1.addWidget(self.show_ci)
        form.addRow(opt_row1)

        self.custom_expr = QComboBox()
        self.custom_expr.setEditable(True)
        self.custom_expr.setPlaceholderText("Custom: e.g. a*x**2 + b*x + c")
        form.addRow(QLabel("Custom expr:"), self.custom_expr)

        # Buttons
        btn_row = QHBoxLayout()
        self.btn_plot = QPushButton("Plot Data")
        self.btn_fit  = QPushButton("Run Fit")
        self.btn_save = QPushButton("Save PNG")
        btn_row.addWidget(self.btn_plot); btn_row.addWidget(self.btn_fit); btn_row.addWidget(self.btn_save)
        form.addRow(btn_row)

        # Canvas (force visible background)
        self.fig = Figure(facecolor="white")
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        left_lay.addWidget(self.canvas)

        # SpanSelector for interactive range
        self.span = SpanSelector(
            self.ax, self._on_span, direction="horizontal",
            useblit=True, props=dict(alpha=0.2, facecolor="#008CBA"), interactive=True
        )

        # Wiring
        self.btn_plot.clicked.connect(self._plot_scatter)
        self.btn_fit.clicked.connect(self._run_fit)
        self.btn_save.clicked.connect(self._save_png)
        self.x_combo.currentIndexChanged.connect(self._plot_scatter)
        self.y_combo.currentIndexChanged.connect(self._plot_scatter)

        # Draw once
        self._plot_scatter(first=True)

    # ---------- data helpers ----------
    def _get_xy(self) -> Tuple[np.ndarray, np.ndarray, str, str]:
        xcol = self.x_combo.currentText()
        ycol = self.y_combo.currentText()
        if not xcol or not ycol or xcol not in self.df or ycol not in self.df:
            return np.array([]), np.array([]), xcol, ycol
        x = self.df[xcol].to_numpy(dtype=float, copy=False)
        y = self.df[ycol].to_numpy(dtype=float, copy=False)
        return x, y, xcol, ycol

    # ---------- UI actions ----------
    def _plot_scatter(self, first: bool=False):
        x, y, xcol, ycol = self._get_xy()
        self.ax.clear()

        if x.size == 0 or y.size == 0:
            self.ax.text(0.5, 0.5, "Choose valid X and Y numeric columns.",
                         ha="center", va="center", transform=self.ax.transAxes)
            self.canvas.draw_idle()
            return

        # set default range once
        if first:
            finite = np.isfinite(x)
            if np.any(finite):
                lo, hi = float(np.nanmin(x[finite])), float(np.nanmax(x[finite]))
                self.xmin_spin.setValue(lo); self.xmax_spin.setValue(hi)
            finite_y = np.isfinite(y)
            if np.any(finite_y):
                lo, hi = float(np.nanmin(y[finite_y])), float(np.nanmax(y[finite_y]))
                self.ymin_spin.setValue(lo); self.ymax_spin.setValue(hi)

        # Apply ranges
        xmin, xmax = self.xmin_spin.value(), self.xmax_spin.value()
        ymin, ymax = self.ymin_spin.value(), self.ymax_spin.value()
        mask = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)

        xs, ys = x[mask], y[mask]

        self.ax.scatter(xs, ys, s=18, edgecolor="black", linewidth=0.4, color="#00A6ED", label="Data")
        self.ax.set_title(f"{ycol} vs {xcol}")
        self.ax.set_xlabel(xcol); self.ax.set_ylabel(ycol)
        self.ax.legend(loc="best")
        self.fig.tight_layout()
        self.canvas.draw_idle()
        self._write_tip()

    def _write_tip(self):
        self.results.setPlainText(
            "Tip: Drag on the plot to set the fit interval (blue translucent band).\n"
            "Then click ‘Run Fit’. Use the spinboxes to tweak exact x-min/x-max and y-min/y-max.\n"
            "Custom model uses variables: x and letters a, b, c, ... in order."
        )

    def _on_span(self, xmin: float, xmax: float):
        self.xmin_spin.setValue(float(min(xmin, xmax)))
        self.xmax_spin.setValue(float(max(xmin, xmax)))
        self.canvas.draw_idle()

    def _save_png(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Save PNG", "fit.png", "PNG Files (*.png)")
        if not fn: return
        if not fn.lower().endswith(".png"): fn += ".png"
        self.fig.savefig(fn, dpi=300, facecolor=self.fig.get_facecolor())

    # ---------- fit core ----------
    def _run_fit(self):
        x, y, xcol, ycol = self._get_xy()
        xmin, xmax = self.xmin_spin.value(), self.xmax_spin.value()
        ymin, ymax = self.ymin_spin.value(), self.ymax_spin.value()

        # Apply mask
        mask = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
        x, y = x[mask], y[mask]

        if len(x) < 3:
            QMessageBox.warning(self, "Not enough data", "No data points in chosen interval.")
            return

        name = self.model.currentText()
        try:
            fit = self._fit_dispatch(name, x, y)
        except Exception as e:
            self.results.setPlainText(f"Fit error: {e}")
            return

        # Draw scatter with range again
        self._plot_scatter(first=False)

        # Overlay fit
        order = np.argsort(fit.x)
        self.ax.plot(fit.x[order], fit.y_fit[order], "--", color="#FF6B6B", lw=2, label=f"{fit.name} fit")

        # CI band if available
        if self.show_ci.isChecked() and fit.cov is not None:
            try:
                sig = np.sqrt(np.diag(fit.cov))
                ci = 1.96 * (np.nanmean(sig) if sig.size > 0 else 0.0)
                self.ax.fill_between(fit.x[order], fit.y_fit[order]-ci, fit.y_fit[order]+ci,
                                     alpha=0.18, color="#FF6B6B", linewidth=0)
            except Exception:
                pass

        self.ax.legend(loc="best")
        self.fig.tight_layout()
        self.canvas.draw_idle()

        # Results text
        lines = [f"Model: {fit.name}"]
        if fit.r2 is not None and not np.isnan(fit.r2):
            lines.append(f"R²: {fit.r2:.5f}")
        for k, v in fit.params.items():
            lines.append(f"{k} = {v:.6g}")
        self.results.setPlainText("\n".join(lines))


    # ---------- dispatch ----------
    def _fit_dispatch(self, name: str, x: np.ndarray, y: np.ndarray) -> FitResult:
        name = name.lower()

        if name.startswith("linear"):
            p, cov = np.polyfit(x, y, 1, cov=True)
            yhat = np.poly1d(p)(x)
            return FitResult(
                name="Linear",
                params={"slope": float(p[0]), "intercept": float(p[1])},
                x=x, y_fit=yhat, cov=cov, r2=_r2_score(y, yhat)
            )

        if name.startswith("polynomial"):
            deg = 2 if "2" in name else 3
            p, cov = np.polyfit(x, y, deg, cov=True)
            yhat = np.poly1d(p)(x)
            params = {f"c{i}": float(v) for i, v in enumerate(p[::-1])}  # c0 + c1 x + ...
            return FitResult(
                name=f"Polynomial (deg {deg})",
                params=params, x=x, y_fit=yhat, cov=cov, r2=_r2_score(y, yhat)
            )

        if name.startswith("exponential"):
            # require positive y for log-likelihood stability
            mask = y > 0
            if np.count_nonzero(mask) < 3:
                raise ValueError("Exponential requires positive y values.")
            popt, pcov = curve_fit(_exp, x[mask], y[mask], p0=(max(y[mask]), 0.1))
            yhat = _exp(x, *popt)
            return FitResult(
                name="Exponential", params={"a": popt[0], "b": popt[1]},
                x=x, y_fit=yhat, cov=pcov, r2=_r2_score(y[mask], _exp(x[mask], *popt))
            )

        if name.startswith("gaussian"):
            p0 = (float(np.nanmax(y)), float(np.nanmean(x)), float(np.nanstd(x)) or 1.0)
            popt, pcov = curve_fit(_gauss, x, y, p0=p0, maxfev=10000)
            yhat = _gauss(x, *popt)
            return FitResult(
                name="Gaussian", params={"a": popt[0], "mu": popt[1], "sigma": popt[2]},
                x=x, y_fit=yhat, cov=pcov, r2=_r2_score(y, yhat)
            )

        if name.startswith("lorentz"):
            p0 = (float(np.nanmax(y)), float(np.nanmean(x)), 1.0)
            popt, pcov = curve_fit(_lorentz, x, y, p0=p0, maxfev=10000)
            yhat = _lorentz(x, *popt)
            return FitResult(
                name="Lorentzian", params={"a": popt[0], "x0": popt[1], "gamma": popt[2]},
                x=x, y_fit=yhat, cov=pcov, r2=_r2_score(y, yhat)
            )

        if name.startswith("voigt"):
            p0 = (float(np.nanmax(y)), float(np.nanmean(x)), float(np.nanstd(x)) or 1.0, 1.0)
            popt, pcov = curve_fit(_voigt, x, y, p0=p0, maxfev=20000)
            yhat = _voigt(x, *popt)
            return FitResult(
                name="Voigt", params={"a": popt[0], "mu": popt[1], "sigma": popt[2], "gamma": popt[3]},
                x=x, y_fit=yhat, cov=pcov, r2=_r2_score(y, yhat)
            )

        if name.startswith("custom"):
            expr = self.custom_expr.currentText().strip()
            if not expr:
                raise ValueError("Enter a custom expression, e.g. a*x**2 + b*x + c")
            # a, b, c, d, e ... plus x available
            letters = list("abcdefghijklmnopqrstuvwxyz")
            def f(x, *vals):
                ns = dict(zip(letters, vals)); ns["x"] = x
                return eval(expr, {}, ns)
            popt, pcov = curve_fit(f, x, y, p0=[1, 1, 1], maxfev=20000)
            yhat = f(x, *popt)
            params = {letters[i]: float(v) for i, v in enumerate(popt)}
            return FitResult(
                name=f"Custom: {expr}", params=params, x=x, y_fit=yhat, cov=pcov, r2=_r2_score(y, yhat)
            )

        raise ValueError(f"Unknown model: {name}")


# ──────────────────────────────────────────────────────────────────────────────
# Wrapper window that your app already uses (openFitWindow → FitWindow)
# ──────────────────────────────────────────────────────────────────────────────
class FitWindow(QMainWindow):
    def __init__(self, df: pd.DataFrame, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Fit Data")
        self.resize(1000, 720)

        self.panel = FitDialog(df, self)
        self.setCentralWidget(self.panel)

# —————————————————————————————————
#  TERMINAL DOCK (Embedded Python Console)
# —————————————————————————————————

from PyQt5.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QPlainTextEdit,
    QPushButton, QSizePolicy, QTextBrowser
)
from PyQt5.QtCore import Qt, QEvent
import traceback, json, os, io
import matplotlib.pyplot as plt


class TerminalDock(QDockWidget):
    """
    Embedded Python console that lets users run custom code against `df`.
    Provides a REPL-like interface with inline results, syntax colors,
    history persistence, and support for pandas/numpy.
    """
    HISTORY_FILE = "terminal_history.json"

    def __init__(self, parent=None, model=None):
        super().__init__("Terminal", parent)
        self.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.TopDockWidgetArea)
        self.setFeatures(QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable)

        self.model = model  # PandasModel for updating the DataFrame
        self.history = []
        self.exec_context = {}  # Persistent namespace across runs

        # ─── Main container ───
        container = QWidget()
        layout = QVBoxLayout(container)

        # Input box
        self.console = QPlainTextEdit()
        self.console.setPlaceholderText(
            ">>> Type Python code here (use df, pd, np)\n"
            "Shift+Enter = newline, Ctrl+Enter = run"
        )
        self.console.setStyleSheet("""
            QPlainTextEdit {
                background: #0D0D0D;
                color: #00FF00;
                font-family: Consolas, monospace;
                font-size: 13px;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 6px;
            }
        """)
        self.console.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.console.installEventFilter(self)
        layout.addWidget(self.console)

        # Output area
        self.output = QTextBrowser()
        self.output.setOpenExternalLinks(True)
        self.output.setStyleSheet("""
            QTextBrowser {
                background: #111;
                color: #EEE;
                font-family: Consolas, monospace;
                font-size: 13px;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 6px;
            }
        """)
        layout.addWidget(self.output)

        # Run button
        run_button = QPushButton("▶ Run Code (Ctrl+Enter)")
        run_button.setStyleSheet("""
            QPushButton {
                background: #008CBA;
                color: white;
                border-radius: 4px;
                padding: 6px 12px;
            }
            QPushButton:hover { background: #0072A3; }
        """)
        run_button.clicked.connect(self.execute_code)
        layout.addWidget(run_button)

        self.setWidget(container)

        # Load history
        self._load_history()

        # Show banner
        self._append_output(
            "Terminal ready.\nYou can use `df` (current data), `pd` (pandas), `np` (numpy).\n"
            "Shift+Enter = newline, Ctrl+Enter = run",
            "cyan"
        )

    # ─── Persistence ───
    def _load_history(self):
        if os.path.exists(self.HISTORY_FILE):
            try:
                with open(self.HISTORY_FILE, "r") as f:
                    self.history = json.load(f)
            except Exception:
                self.history = []

    def _save_history(self):
        try:
            with open(self.HISTORY_FILE, "w") as f:
                json.dump(self.history, f)
        except Exception:
            pass

    # ─── Append Output ───
    def _append_output(self, text, color="white"):
        self.output.append(f"<pre style='color:{color}'>{text}</pre>")

    # ─── Execute Code ───
    def execute_code(self):
        code = self.console.toPlainText().strip()
        if not code:
            return

        self.history.append(code)
        self._save_history()

        try:
            # Provide a persistent namespace
            local_vars = {"df": getattr(self.model, "df", None)}
            global_vars = {"pd": __import__("pandas"), "np": __import__("numpy")}

            last_expr = compile(code, "<terminal>", "exec")
            exec(last_expr, global_vars, local_vars)

            # Success message
            self._append_output("[✔] Code executed successfully", "lightgreen")

            # If user modified df, update model
            if "df" in local_vars and local_vars["df"] is not None:
                self.model.df = local_vars["df"]
                self.model.layoutChanged.emit()

        except Exception:
            tb = traceback.format_exc()
            self._append_output(tb, "red")

    # ─── Event Filter for Ctrl+Enter ───
    def eventFilter(self, obj, event):
        if obj is self.console and event.type() == QEvent.KeyPress:
            if (event.modifiers() & Qt.ControlModifier) and event.key() == Qt.Key_Return:
                self.execute_code()
                return True
            elif (event.modifiers() & Qt.ShiftModifier) and event.key() == Qt.Key_Return:
                cursor = self.console.textCursor()
                cursor.insertText("\n")
                return True
        return super().eventFilter(obj, event)
    

import os
from gpt4all import GPT4All

# ──────────────────────────────────────────────────────────────────────────────
# Dataspec Assistant Dock — robust + safe LLM fallback
# ──────────────────────────────────────────────────────────────────────────────
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDockWidget, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTextBrowser, QLineEdit, QPushButton, QMessageBox, QCheckBox, QFileDialog,
    QApplication
)
import os, io, textwrap, difflib
import numpy as np
import pandas as pd

try:
    from gpt4all import GPT4All
except Exception:
    GPT4All = None


class AITabDock(QDockWidget):
    """
    Two-tab assistant:
      • Dataspec Assistant — answers from HELP_KB
      • AI Analysis — native analytics + optional GPT4All
    Robust against GPT4All context errors; always returns *something* useful.
    """
    _MAX_CHARS_PROMPT = 1800         # hard cap for LLM prompt text
    _MAX_TOKENS = 256                # generation cap
    _TOP_CORR_PAIRS = 12
    _MAX_SHOW_COLS = 20
    _MAX_SHOW_TYPES = 20
    _MAX_SHOW_MISSING = 20

    def __init__(self, parent, model):
        super().__init__("Dataspec Assistant", parent)
        self.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self.setFeatures(QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable)

        self.model = model
        self.llm = None

        # Try to load a local GPT4All model (if present)
        if GPT4All is not None:
            for name in (
                "mistral-7b-instruct-v0.1.Q4_0.gguf",
                "ggml-gpt4all-j-v1.3-groovy",
            ):
                path = os.path.expanduser(f"~/.cache/gpt4all/{name}")
                if os.path.exists(path):
                    try:
                        self.llm = GPT4All(path)
                        break
                    except Exception:
                        self.llm = None

        tabs = QTabWidget()
        tabs.addTab(self._build_help_tab(), "Dataspec Assistant")
        tabs.addTab(self._build_dataset_ai_tab(), "AI Analysis")
        self.setWidget(tabs)

    # ============================ HELP TAB ============================
    def _build_help_tab(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        blurb = QLabel(
            "💡 Ask about any Dataspec feature or toolbar button.\n"
            "Example: “What does Save Workflow do?”"
        )
        blurb.setWordWrap(True)
        lay.addWidget(blurb)

        self.help_display = QTextBrowser()
        lay.addWidget(self.help_display)

        row = QHBoxLayout()
        self.help_input = QLineEdit()
        self.help_input.setPlaceholderText("Ask about a feature (e.g., Profile Report, Terminal, Fit)...")
        btn = QPushButton("Ask")
        btn.clicked.connect(self.ask_help)
        row.addWidget(self.help_input); row.addWidget(btn)
        lay.addLayout(row)
        return w

    def ask_help(self):
        q = self.help_input.text().strip()
        if not q: return
        self.help_display.append(f"<b>You:</b> {q}")
        try:
            self.help_display.append(f"<b>Assistant:</b> {self._kb_answer(q)}")
        except Exception as e:
            self.help_display.append(f"<b>Assistant:</b> Error reading help: {e}")
        self.help_input.clear()

    def _kb_answer(self, q: str) -> str:
        ql = q.lower()
        kb = globals().get("HELP_KB", {})
        toolbar = kb.get("toolbar", {})

        # direct key or title match
        for key, item in toolbar.items():
            title = item.get("title", "")
            if key in ql or title.lower() in ql:
                return self._fmt_kb_item(item)

        # fuzzy title/key match
        best = None
        best_score = 0.0
        for key, item in toolbar.items():
            title = item.get("title", "")
            s = max(
                difflib.SequenceMatcher(None, ql, key.lower()).ratio(),
                difflib.SequenceMatcher(None, ql, title.lower()).ratio(),
            )
            if s > best_score:
                best, best_score = item, s
        if best and best_score > 0.45:
            return self._fmt_kb_item(best)

        return "❓ I couldn’t find that feature."

    @staticmethod
    def _fmt_kb_item(item: dict) -> str:
        return (
            f"<b>{item.get('title','Feature')}</b><br>"
            f"{item.get('description','')}<br>"
            f"<i>Example:</i> {item.get('example','')}"
        )

    # ============================ AI TAB ============================
    def _build_dataset_ai_tab(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)

        head = QLabel(
            "🤖 <b>AI Data Consultant</b><br>"
            "Ask natural-language questions about your dataset "
            "(summary, correlations, missing values, trends, anomalies, etc.)."
        )
        head.setWordWrap(True)
        lay.addWidget(head)

        self.chat = QTextBrowser()
        lay.addWidget(self.chat)

        # Controls row
        row1 = QHBoxLayout()
        self.toggle_llm = QCheckBox("Enable GPT4All")
        self.toggle_llm.setChecked(True if self.llm else False)
        copy_btn = QPushButton("Copy")
        copy_btn.clicked.connect(self._copy_chat)
        export_btn = QPushButton("Export")
        export_btn.clicked.connect(self._export_chat)
        row1.addWidget(self.toggle_llm); row1.addStretch(); row1.addWidget(copy_btn); row1.addWidget(export_btn)
        lay.addLayout(row1)

        # Input
        row2 = QHBoxLayout()
        self.q_input = QLineEdit()
        self.q_input.setPlaceholderText("e.g., What columns are correlated?")
        btn = QPushButton("Ask AI")
        btn.clicked.connect(self.ask_dataset_ai)
        row2.addWidget(self.q_input); row2.addWidget(btn)
        lay.addLayout(row2)

        return w

    def ask_dataset_ai(self):
        q = self.q_input.text().strip()
        if not q:
            return
        self.chat.append(f"<b>You:</b> {q}")

        df = self.model.getDataFrame()
        if df is None or df.empty:
            self.chat.append("<b>AI:</b> ⚠️ No dataset loaded.")
            self.q_input.clear()
            return

        # Try native answers first
        if self._maybe_answer_natively(df, q):
            self.q_input.clear()
            return  # ✅ Stop here if handled natively

        # Otherwise, try LLM if enabled
        if self.toggle_llm.isChecked() and self.llm:
            ans = self._llm_answer_safe(df, q)
            self.chat.append("<b>AI:</b> " + self._escape(ans))
        else:
            self.chat.append("<b>AI:</b> GPT4All is disabled or unavailable.")
        self.q_input.clear()

    # ---------- Native answers (no LLM) ----------
    def _maybe_answer_natively(self, df: pd.DataFrame, q: str) -> bool:
        ql = q.lower()

        def reply(text, code=None, as_pre=True):
            if as_pre:
                self.chat.append("<b>AI:</b><pre>" + self._escape(text) + "</pre>")
            else:
                self.chat.append("<b>AI:</b> " + self._escape(text))
            if code:
                self.chat.append(f"<i>Try in Terminal:</i> <code>{code}</code>")

        # keyword families
        if any(w in ql for w in ("summary", "overview", "describe", "stats", "information", "shape")):
            reply(self._summary_text(df), "df.describe(include='all')")
            return True

        if any(w in ql for w in ("correl", "correlated", "correlation", "pearson")):
            reply(self._correlation_text(df), "df.corr()")
            return True

        if any(w in ql for w in ("missing", "null", "nan", "na")):
            reply(self._missing_text(df), "df.isna().sum()")
            return True

        if any(w in ql for w in ("head", "preview", "sample", "first rows")):
            reply(df.head(10).to_string(), "df.head(10)")
            return True

        if any(w in ql for w in ("dtype", "types", "schema", "columns")):
            info = pd.DataFrame({"dtype": df.dtypes.astype(str)}) \
                    .head(self._MAX_SHOW_TYPES).to_string()
            reply(info, "df.dtypes")
            return True

        if any(w in ql for w in ("unique", "distinct", "categories", "levels")):
            out = io.StringIO()
            for c in df.columns[:self._MAX_SHOW_COLS]:
                try:
                    u = df[c].nunique()
                    print(f"{c}: {u} unique", file=out)
                except Exception:
                    pass
            reply(out.getvalue(), "df.nunique()")
            return True

        if any(w in ql for w in ("outlier", "anomal")):
            num = df.select_dtypes(include="number")
            if num.empty:
                reply("No numeric columns to compute outliers.")
                return True
            z = ((num - num.mean()) / (num.std(ddof=0)+1e-9)).abs()
            hits = (z > 3).sum().sort_values(ascending=False).head(10).to_string()
            reply("Top columns by # of z-score>3 outliers:\n" + hits)
            self.chat.append("<i>Try in Terminal:</i> <code>((df.select_dtypes('number')-df.mean())/df.std()).abs()>3</code>")
            return True

        if any(w in ql for w in ("trend", "time series", "ts", "rolling", "moving average")):
            reply(
                "Look for a datetime column, then compute a rolling mean:\n"
                "1) Set index: df = df.set_index('Date')\n"
                "2) df['y'].rolling(7).mean()\n"
                "3) Plot in Dashboard.",
                "df['y'].rolling(7).mean()"
            )
            return True

        if any(w in ql for w in ("histogram", "distribution", "kde", "density")):
            reply(
                "Use Explore → Histogram/KDE. Programmatically:\n"
                "df['col'].hist(bins=30)  # or seaborn.kdeplot(df['col'])",
                "df['col'].hist(bins=30)"
            )
            return True

        # No native route -> not handled
        return False

    # ---------- Native helpers ----------
    def _summary_text(self, df: pd.DataFrame) -> str:
        buf = io.StringIO()
        print(f"Shape: {df.shape[0]} rows × {df.shape[1]} cols", file=buf)
        cols = ", ".join(list(map(str, df.columns[:self._MAX_SHOW_COLS])))
        print(f"Columns: {cols}", file=buf)
        try:
            desc = df.describe(include="all").transpose().head(12)
            print(desc.to_string(), file=buf)
        except Exception as e:
            print(f"(describe failed: {e})", file=buf)
        return buf.getvalue()

    def _correlation_text(self, df: pd.DataFrame) -> str:
        num = df.select_dtypes(include="number")
        if num.shape[1] < 2:
            return "Not enough numeric columns to compute correlations."
        corr = num.corr()
        # upper triangle pairs sorted
        pairs = corr.where(~np.tril(np.ones(corr.shape)).astype(bool)) \
                    .stack().sort_values(ascending=False)
        return pairs.head(self._TOP_CORR_PAIRS).to_string()

    def _missing_text(self, df: pd.DataFrame) -> str:
        miss = df.isna().sum()
        miss = miss[miss > 0]
        return "No missing values." if miss.empty else miss.head(self._MAX_SHOW_MISSING).to_string()

    # ---------- LLM with hard safety ----------
    def _llm_answer_safe(self, df: pd.DataFrame, question: str) -> str:
        """
        Build a *tiny* prompt and always catch GPT4All errors.
        If anything goes wrong, return a helpful fallback string.
        """
        meta = f"Shape: {df.shape[0]} rows, {df.shape[1]} cols. " \
               f"Columns: {', '.join(map(str, df.columns[:10]))}"
        prompt = (
            "You are a data analyst. Given the dataset metadata and a question, "
            "answer concisely and practically.\n\n"
            f"Dataset: {meta}\n\nQuestion: {question}\nAnswer:"
        )
        if len(prompt) > self._MAX_CHARS_PROMPT:
            prompt = prompt[:self._MAX_CHARS_PROMPT]

        try:
            # Start a fresh, short chat each time
            with self.llm.chat_session():
                return self.llm.generate(
                    prompt,
                    max_tokens=self._MAX_TOKENS,
                    temp=0.3,
                    top_k=40,
                    top_p=0.9,
                )
        except Exception as e:
            # Fallback if context too large or any other engine error
            return (
                "I couldn’t use the AI engine just now. "
                "Here are quick next steps you can try:\n"
                "• Summary: df.describe(include='all')\n"
                "• Correlations: df.corr()\n"
                "• Missing values: df.isna().sum()\n"
                "• Preview: df.head(10)"
            )

    # ---------- utility ----------
    def _copy_chat(self):
        txt = self.chat.toPlainText()
        if not txt: return
        QApplication.clipboard().setText(txt)
        QMessageBox.information(self, "Copied", "Answer copied to clipboard.")

    def _export_chat(self):
        fn, _ = QFileDialog.getSaveFileName(
            self, "Export Answer", "assistant_output.txt",
            "Text Files (*.txt);;HTML Files (*.html)"
        )
        if not fn: return
        txt = self.chat.toPlainText()
        if fn.lower().endswith(".html"):
            txt = "<pre>" + self._escape(txt) + "</pre>"
        with open(fn, "w", encoding="utf-8") as f:
            f.write(txt)
        QMessageBox.information(self, "Exported", f"Answer exported to {fn}")

    @staticmethod
    def _escape(s: str) -> str:
        # very light escaping for QTextBrowser
        return textwrap.dedent(str(s)).strip().replace("<", "&lt;").replace(">", "&gt;")


HELP_KB = {
    "toolbar": {
        "load_file": {
            "title": "Load File",
            "description": (
                "Open datasets into Dataspec. Supports CSV, TXT, and DAT formats. "
                "Large files (>10MB) are loaded in chunks with progress tracking."
            ),
            "example": "Click 'Load File' (Ctrl+O) → choose dataset.csv."
        },
        "save_file": {
            "title": "Save File",
            "description": (
                "Save the current DataFrame to CSV, Excel, or JSON. "
                "Ensures your cleaned data is exportable for future work."
            ),
            "example": "Click 'Save File' (Ctrl+S) → save as cleaned_data.csv."
        },
        "drop_na": {
            "title": "Drop NA Rows",
            "description": (
                "Removes all rows with missing values. Useful when incomplete "
                "records could distort analysis."
            ),
            "example": "Click 'Drop NA Rows' → dataset shrinks to only complete rows."
        },
        "fill_na": {
            "title": "Fill NA",
            "description": (
                "Replaces missing values with a strategy: Mean, Median, Forward Fill, "
                "Backward Fill, or a user-defined Constant."
            ),
            "example": "Choose 'Fill NA' → 'Mean' → all blanks replaced by column averages."
        },
        "remove_selected": {
            "title": "Remove Selected Rows",
            "description": (
                "Deletes only the rows currently selected in the table view. "
                "Allows precise cleaning."
            ),
            "example": "Select rows 5–10 → 'Remove Selected' → they disappear."
        },
        "rename_column": {
            "title": "Rename Column",
            "description": (
                "Change a column name for clarity, readability, or consistency."
            ),
            "example": "Rename 'col_1' → 'Customer_ID'."
        },
        "delete_column": {
            "title": "Delete Column",
            "description": (
                "Removes an entire column from the dataset permanently."
            ),
            "example": "Delete 'Temp_Notes' column if not needed."
        },
        "conditional_format": {
            "title": "Conditional Formatting",
            "description": (
                "Highlight cells that match rules (>, <, ==, !=, contains). "
                "Great for spotting outliers or flagged conditions."
            ),
            "example": "Highlight rows where 'Revenue' > 1,000,000 in green."
        },
        "fit": {
            "title": "Fit Data",
            "description": (
                "Apply advanced curve fitting: Linear, Polynomial, Exponential, "
                "Gaussian, Lorentzian, Voigt, Custom equations, and Bayesian Linear regression."
            ),
            "example": "Fit a Gaussian curve to a peak dataset to extract mean and sigma."
        },
        "visualize": {
            "title": "Visualize",
            "description": (
                "Quick histograms of numeric columns. Helps understand distribution shape, "
                "skewness, and spread."
            ),
            "example": "Visualize 'Age' column → see bell curve distribution."
        },
        "dashboard": {
            "title": "Dashboard",
            "description": (
                "Explore correlations, box plots, and scatter plots across multiple tabs. "
                "Designed for interactive exploratory data analysis."
            ),
            "example": "Open Dashboard → check correlation heatmap between 'Income' and 'Spending'."
        },
        "profile_report": {
            "title": "Profile Report",
            "description": (
                "Generates a full HTML report (summary stats, missing values, distributions, "
                "correlations, duplicates). Requires ydata-profiling installed."
            ),
            "example": "Click 'Profile Report' → get auto-generated insights in an HTML file."
        },
        "save_workflow": {
            "title": "Save Workflow",
            "description": (
                "Export all cleaning steps into a Python script for reproducibility. "
                "Lets you re-run the same process later."
            ),
            "example": "Save workflow → produces a .py file with all steps applied."
        },
        "load_workflow": {
            "title": "Load Workflow",
            "description": (
                "Apply a previously exported workflow (.py) to new data. "
                "Ensures consistency across datasets."
            ),
            "example": "Load workflow 'cleaning_steps.py' → applies to new dataset."
        },
        "undo": {
            "title": "Undo",
            "description": "Reverts the most recent cleaning action (supports up to 10 undo steps).",
            "example": "Deleted a column by mistake? Press Undo (Ctrl+Z)."
        },
        "terminal": {
            "title": "Terminal",
            "description": (
                "Embedded Python REPL with pandas/numpy preloaded. "
                "Write and run Python code directly against your dataset."
            ),
            "example": "Type `df.head()` to preview the top rows."
        },
        "ai_assistant": {
            "title": "AI Assistant",
            "description": (
                "Offline AI (GPT4All) that analyzes your dataset. "
                "Ask natural-language questions like 'Summarize sales by region'."
            ),
            "example": "Ask: 'What columns are highly correlated?'"
        },
        "dataspec_assistant": {
            "title": "Dataspec Assistant",
            "description": (
                "Knows Dataspec inside and out. Explains toolbar buttons, shows terminal examples, "
                "and gives advanced usage tips."
            ),
            "example": "Ask: 'What does Conditional Formatting do?'"
        },
        "settings": {
            "title": "Settings",
            "description": (
                "Review software help, shortcuts, and preferences. "
                "Customize keyboard shortcuts and appearance (dark/light)."
            ),
            "example": "Change 'Toggle Dark/Light' from Ctrl+T → Alt+D."
        },
        "toggle": {
            "title": "Dark/Light Mode",
            "description": "Switch between dark mode (high contrast) and light mode (clean white).",
            "example": "Click 'Light Mode' → instantly switch to light theme."
        },
        "home": {
            "title": "Home",
            "description": (
                "Return to the welcome screen with Dataspec logo, matrix animation, "
                "and usage instructions."
            ),
            "example": "Click 'Home' to reset view before loading a new dataset."
        }
    },

    "terminal_examples": [
        {"task": "Preview dataset", "code": "df.head()"},
        {"task": "Check shape", "code": "df.shape"},
        {"task": "Summary stats", "code": "df.describe(include='all')"},
        {"task": "Filter rows", "code": "df[df['Sales'] > 1000]"},
        {"task": "Add new column", "code": "df['Total'] = df['Price'] * df['Quantity']"},
        {"task": "Group & aggregate", "code": "df.groupby('Region')['Revenue'].sum()"},
        {"task": "Sort values", "code": "df.sort_values('Date', ascending=False)"},
        {"task": "Missing values per column", "code": "df.isna().sum()"},
        {"task": "Drop column", "code": "df.drop(columns=['Unused'])"},
        {"task": "Correlation matrix", "code": "df.corr()"}
    ],

    "advanced_tips": [
        "💡 Use the Fit Data window to model non-linear relationships like Gaussian or Voigt peaks.",
        "💡 The Dashboard is ideal for EDA: check correlations and distributions side by side.",
        "💡 Save & Load Workflows to make your cleaning process reproducible.",
        "💡 The Terminal is as powerful as a Jupyter cell—anything pandas/numpy can do, you can do here.",
        "💡 Profile Reports (ydata-profiling) generate quick exploratory reports for presentation.",
        "💡 Conditional Formatting makes outliers pop instantly.",
        "💡 Use the AI Assistant to get quick statistical insights, and the Dataspec Assistant to learn the software itself."
    ]
}

class DataspecAssistantDock(QDockWidget):
    def __init__(self, parent=None):
        super().__init__("Dataspec Assistant", parent)
        self.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self.setFeatures(QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable)

        # Load GPT4All model (offline)
        model_filename = "mistral-7b-instruct-v0.1.Q4_0.gguf"
        model_path = os.path.expanduser(f"~/.cache/gpt4all/{model_filename}")
        self.llm = GPT4All(model_path)

        # UI setup
        container = QWidget()
        layout = QVBoxLayout(container)

        self.chat_display = QTextBrowser()
        layout.addWidget(self.chat_display)

        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("Ask about Dataspec features...")
        layout.addWidget(self.input_box)

        send_btn = QPushButton("Ask Assistant")
        send_btn.clicked.connect(self.ask_assistant)
        layout.addWidget(send_btn)

        self.setWidget(container)

    def ask_assistant(self):
        question = self.input_box.text().strip()
        if not question:
            return

        # Match question to KB keywords
        kb_snippet = self._find_help(question)

        # Build prompt
        prompt = f"""
        You are the Dataspec Assistant. You know every feature of the software.
        User asked: "{question}"
        Reference info:
        {kb_snippet}

        Answer clearly and helpfully, with examples if relevant.
        """

        with self.llm.chat_session():
            response = self.llm.generate(prompt, max_tokens=400)

        self.chat_display.append(f"<b>You:</b> {question}")
        self.chat_display.append(f"<b>Assistant:</b> {response}")
        self.input_box.clear()

    def _find_help(self, question: str) -> str:
        q = question.lower()
        # Look in toolbar KB
        for key, val in HELP_KB["toolbar"].items():
            if key in q or val["title"].lower() in q:
                return f"{val['title']}: {val['description']} Example: {val['example']}"
        # Check terminal examples
        if "terminal" in q or "example" in q:
            examples = "\n".join([f"{ex['task']}: {ex['code']}" for ex in HELP_KB["terminal_examples"]])
            return f"Terminal Examples:\n{examples}"
        # Otherwise return advanced tips
        return "\n".join(HELP_KB["advanced_tips"])


# —————————————————————————————————
#  WELCOME WIDGET (Matrix + Programmatic Logo + Text)
# —————————————————————————————————

class WelcomeWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Draw our “matrix” behind everything
        self.setAttribute(Qt.WA_StyledBackground, False)

        # Monospaced font for falling hex digits
        self.matrix_font = QFont("Courier", 10)
        self.setFont(self.matrix_font)

        # Two gray bases to fade text
        self.gray_base_dark = QColor(80, 80, 80)
        self.gray_base_light = QColor(200, 200, 200)

        # Characters to draw
        self.text_chars = "0123456789ABCDEF"

        # Timer drives ~5 frames per second
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateMatrix)
        self.timer.start(200)

        # Compute initial char sizes
        fm = QFontMetrics(self.matrix_font)
        self.char_w = fm.horizontalAdvance("0")
        self.char_h = fm.height()
        self.columns = max(1, self.width() // self.char_w)
        self.rows = max(1, self.height() // self.char_h)

        # Each column’s “y” starts above top
        self.y_positions = [random.randint(-self.rows, 0) for _ in range(self.columns)]

        # ─────────────────────────────────────────────
        # Layout: custom LogoWidget + instructions
        # ─────────────────────────────────────────────
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        layout.addStretch()

        # ← This was here, before the logo:
        layout.addSpacing(240)

        # Inject our LogoWidget
        self.logo_widget = LogoWidget(self)
        self.logo_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.logo_widget)

        # Add some spacing between logo+title and instructions
        layout.addSpacing(100)

        # Instruction QLabel
        self.instr_label = QLabel()
        self.instr_label.setAlignment(Qt.AlignCenter)
        self.instr_label.setWordWrap(True)
        self.instr_label.setAttribute(Qt.WA_TranslucentBackground, True)
        self.instr_label.setStyleSheet("background: transparent;")
        layout.addWidget(self.instr_label)

        layout.addStretch()

        # Build the HTML instructions (no color:inherit—just plain <p>):
        self._buildInstructionText()

        # Default to dark‐mode logo + white text
        # (DataCleaningApp.applyStyle will call this again immediately after)
        self.updateLogoAndTextColors(is_dark=True)

        # Fade‐in animation for logo + instructions
        for w in (self.logo_widget, self.instr_label):
            effect = QGraphicsOpacityEffect(w)
            w.setGraphicsEffect(effect)
            anim = QPropertyAnimation(effect, b"opacity", self)
            anim.setDuration(800)
            anim.setStartValue(0.0)
            anim.setEndValue(1.0)
            anim.start()


    def _buildInstructionText(self):
        html = (
        "<p style='font-size:18px; font-family:\"Segoe UI\", sans-serif;'>"
        "<b>Welcome to Dataspec</b> — your all-in-one tool for "
        "data <b>cleaning</b>, <b>analysis</b>, and <b>reporting</b>.<br><br>"

        "Get started by clicking <b>Load File (Ctrl+O)</b> or dropping in a CSV, TXT, or DAT file.<br><br>"

        "<b>Typical workflow:</b><br>"
        "&nbsp;&nbsp;• <b>Clean</b>: Drop or fill missing values, rename/delete columns, remove rows, and apply conditional formatting.<br>"
        "&nbsp;&nbsp;• <b>Analyze</b>: Explore your data with interactive plots, histograms, scatter/box plots, correlation heatmaps, and advanced fitting (linear, polynomial, Gaussian, Voigt, etc.).<br>"
        "&nbsp;&nbsp;• <b>AI Assist</b>: Use the built-in offline AI assistant to ask natural-language questions about your dataset and get instant insights.<br>"
        "&nbsp;&nbsp;• <b>Report</b>: Generate automated HTML profile reports (via <i>ydata-profiling</i>) or export your full cleaning workflow as a Python script.<br><br>"

        "All tools are available in the <b>toolbar</b> above. "
        "Switch between <b>Light/Dark</b> themes, view the <b>Dashboard</b>, or open the <b>Terminal</b> for advanced scripting.<br>"
        "</p>"
        )
        self.instr_label.setText(html)

    def updateLogoAndTextColors(self, is_dark: bool):
        """
        Force LogoWidget to repaint (it now looks at self.window().is_dark_mode).
        Also change instruction text color.
        """
        self.logo_widget.update()
        if is_dark:
            self.instr_label.setStyleSheet("color: #FFFFFF; background: transparent;")
        else:
            self.instr_label.setStyleSheet("color: #000000; background: transparent;")

    def resizeEvent(self, event):
        """
        Recompute how many matrix chars fit whenever the widget is resized.
        """
        fm = QFontMetrics(self.matrix_font)
        self.char_w = fm.horizontalAdvance("0")
        self.char_h = fm.height()
        self.columns = max(1, self.width() // self.char_w)
        self.rows = max(1, self.height() // self.char_h)

        old_len = len(self.y_positions)
        if self.columns > old_len:
            for _ in range(self.columns - old_len):
                self.y_positions.append(random.randint(-self.rows, 0))
        elif self.columns < old_len:
            self.y_positions = self.y_positions[: self.columns]

        super().resizeEvent(event)

    def updateMatrix(self):
        """
        Move each column’s head downward; wrap to top when it passes bottom.
        Then trigger repaint() so that matrix is redrawn behind.
        """
        for i in range(self.columns):
            self.y_positions[i] += 1
            if self.y_positions[i] > self.rows:
                self.y_positions[i] = random.randint(-self.rows, 0)
        self.update()

    def paintEvent(self, event):
        """
        Paint the background (dark or light) and the falling hex digits (“matrix”).
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Determine dark vs. light from self.window().is_dark_mode
        is_dark = False
        main_win = self.window()
        if hasattr(main_win, "is_dark_mode"):
            is_dark = main_win.is_dark_mode

        # Fill background behind matrix:
        if is_dark:
            painter.fillRect(self.rect(), QColor("#0D0D0D"))
            base = self.gray_base_dark
        else:
            painter.fillRect(self.rect(), Qt.white)
            base = self.gray_base_light

        painter.setFont(self.matrix_font)
        fm = QFontMetrics(self.matrix_font)
        cw, ch = self.char_w, self.char_h
        tail = max(1, self.rows // 4)

        for col in range(self.columns):
            head = self.y_positions[col]
            x = col * cw
            for t in range(tail):
                ri = head - t
                if 0 <= ri < self.rows:
                    fade = 1.0 - (t / tail)
                    alpha = int(30 + fade * 170)
                    c = QColor(base)
                    c.setAlpha(alpha)
                    painter.setPen(c)
                    ch_char = random.choice(self.text_chars)
                    y = ri * ch + fm.ascent()
                    painter.drawText(x, y, ch_char)

        painter.end()
        super().paintEvent(event)


import os
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QTextBrowser,
    QLineEdit, QPushButton
)
from PyQt5.QtCore import Qt
from gpt4all import GPT4All
import difflib

# Make sure HELP_KB is defined globally in your project
# (the one you pasted earlier)
# from help_kb import HELP_KB   # if stored separately

class AssistantWindow(QMainWindow):
    """
    Unified Assistant Window for Dataspec.
    Tab 1: Dataspec Assistant (HELP_KB-driven).
    Tab 2: AI Analysis (GPT4All-driven).
    """

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Dataspec Assistant")
        self.setGeometry(200, 200, 750, 600)

        self.model = model  # PandasModel

        # Load GPT4All model (offline AI)
        model_filename = "mistral-7b-instruct-v0.1.Q4_0.gguf"
        model_path = os.path.expanduser(f"~/.cache/gpt4all/{model_filename}")
        self.llm = GPT4All(model_path if os.path.exists(model_path) else model_filename)

        # --- Tabs ---
        tabs = QTabWidget()
        self.setCentralWidget(tabs)

        # 1) Dataspec Assistant tab
        self.ds_tab = QWidget()
        self.ds_layout = QVBoxLayout(self.ds_tab)

        self.ds_intro = QLabel(
            "💡 <b>Dataspec Assistant</b><br>"
            "Ask me about any feature (toolbar button, shortcut, workflow).<br>"
            "Type the name of a button or function, and I’ll explain what it does "
            "with examples and tips."
        )
        self.ds_intro.setWordWrap(True)
        self.ds_layout.addWidget(self.ds_intro)

        self.ds_display = QTextBrowser()
        self.ds_layout.addWidget(self.ds_display)

        self.ds_input = QLineEdit()
        self.ds_input.setPlaceholderText("e.g. Conditional Formatting, Terminal, Fit Data...")
        self.ds_layout.addWidget(self.ds_input)

        ds_btn = QPushButton("Ask Dataspec Assistant")
        ds_btn.clicked.connect(self.ask_dataspec)
        self.ds_layout.addWidget(ds_btn)

        tabs.addTab(self.ds_tab, "Dataspec Assistant")

        # 2) AI Analysis tab
        self.ai_tab = QWidget()
        self.ai_layout = QVBoxLayout(self.ai_tab)

        self.ai_intro = QLabel(
            "🤖 <b>AI Data Consultant</b><br>"
            "Ask me natural-language questions about your dataset "
            "(summary, correlations, trends, anomalies, etc.)."
        )
        self.ai_intro.setWordWrap(True)
        self.ai_layout.addWidget(self.ai_intro)

        self.ai_display = QTextBrowser()
        self.ai_layout.addWidget(self.ai_display)

        self.ai_input = QLineEdit()
        self.ai_input.setPlaceholderText("e.g. What columns are correlated?")
        self.ai_layout.addWidget(self.ai_input)

        ai_btn = QPushButton("Ask AI")
        ai_btn.clicked.connect(self.ask_ai)
        self.ai_layout.addWidget(ai_btn)

        tabs.addTab(self.ai_tab, "AI Analysis")

    # --- Dataspec Assistant ---
    def ask_dataspec(self):
        q = self.ds_input.text().strip().lower()
        if not q:
            return
        answer = None
        for key, entry in HELP_KB["toolbar"].items():
            if q in key or q in entry["title"].lower():
                answer = f"<b>{entry['title']}</b><br>{entry['description']}<br><i>Example:</i> {entry['example']}"
                break
        if not answer:
            answer = "❓ I don’t recognize that feature. Try asking about toolbar buttons like 'Fit Data', 'Terminal', or 'Profile Report'."
        self.ds_display.append(f"<b>You:</b> {q}")
        self.ds_display.append(f"<b>Assistant:</b> {answer}")
        self.ds_input.clear()

    # --- AI Assistant ---
    def ask_ai(self):
        q = self.ai_input.text().strip()
        if not q:
            return
        df = self.model.getDataFrame()
        if df.empty:
            df_summary = "No data loaded."
        else:
            df_summary = df.describe(include="all").to_string()

        prompt = f"Here is the dataset summary:\n{df_summary}\n\nQuestion: {q}\nAnswer clearly and concisely."
        with self.llm.chat_session():
            resp = self.llm.generate(prompt, max_tokens=400)
        self.ai_display.append(f"<b>You:</b> {q}")
        self.ai_display.append(f"<b>AI:</b> {resp}")
        self.ai_input.clear()

# --------------------------
# Extra Windows
# --------------------------
from PyQt5.QtWidgets import QMainWindow

class TerminalWindow(QMainWindow):
    def __init__(self, model=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Terminal")
        self.setGeometry(200, 200, 900, 500)

        # reuse your existing TerminalDock
        self.dock = TerminalDock(self, model)
        self.setCentralWidget(self.dock)

class AIWindow(QMainWindow):
    def __init__(self, model=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AI Assistant")
        self.setGeometry(280, 280, 900, 600)

        # reuse your existing AIDock as the central widget
        self.panel = AITabDock(self, model)
        self.setCentralWidget(self.panel)

class FitWindow(QMainWindow):
    def __init__(self, df=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Fit Data")
        self.setGeometry(220, 220, 900, 600)

        self.panel = FitDialog(df, self)   # reuse FitDialog as a widget
        self.setCentralWidget(self.panel)


# ─────────────────────────────
# ExploreWindow
# ─────────────────────────────
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QTabWidget, QPushButton,
    QFileDialog, QComboBox, QLabel, QHBoxLayout
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix

class DraggableTabWidget(QTabWidget):
    """Custom TabWidget that allows detachable tabs."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTabsClosable(True)
        self.tabCloseRequested.connect(self.close_tab)
        self.setMovable(True)  # allows reordering within tab bar

    def close_tab(self, index):
        widget = self.widget(index)
        if widget:
            widget.deleteLater()
        self.removeTab(index)

    def mouseDoubleClickEvent(self, event):
        """Detach tab on double-click (could be drag if preferred)."""
        index = self.tabBar().tabAt(event.pos())
        if index >= 0:
            widget = self.widget(index)
            title = self.tabText(index)

            # Remove from tab widget
            self.removeTab(index)

            # Create floating window
            win = QMainWindow()
            win.setWindowTitle(f"Detached: {title}")
            win.setCentralWidget(widget)
            win.resize(700, 500)
            win.show()


class ExploreWindow(QMainWindow):
    def __init__(self, df, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Explore Data")
        self.resize(1100, 750)

        self.df = df

        container = QWidget()
        layout = QVBoxLayout(container)

        # Controls row
        controls = QHBoxLayout()
        self.plot_type = QComboBox()
        self.plot_type.addItems(
            ["Histogram", "Boxplot", "Scatter", "Heatmap", "KDE", "Violin", "Time Series"]
        )
        controls.addWidget(QLabel("Plot Type:"))
        controls.addWidget(self.plot_type)

        self.x_select = QComboBox()
        self.x_select.addItems(self.df.columns)
        self.x_select.setEditable(True)
        controls.addWidget(QLabel("X-axis:"))
        controls.addWidget(self.x_select)

        self.y_select = QComboBox()
        self.y_select.addItems(self.df.columns)
        self.y_select.setEditable(True)
        controls.addWidget(QLabel("Y-axis:"))
        controls.addWidget(self.y_select)

        self.x_label_input = QLineEdit()
        self.y_label_input = QLineEdit()
        self.title_input = QLineEdit()
        controls.addWidget(QLabel("X Label:"))
        controls.addWidget(self.x_label_input)
        controls.addWidget(QLabel("Y Label:"))
        controls.addWidget(self.y_label_input)
        controls.addWidget(QLabel("Title:"))
        controls.addWidget(self.title_input)

        add_btn = QPushButton("➕ Add Plot")
        add_btn.clicked.connect(self.add_plot_tab)
        controls.addWidget(add_btn)

        empty_btn = QPushButton("➕ Empty Tab")
        empty_btn.clicked.connect(self.add_empty_tab)
        controls.addWidget(empty_btn)

        layout.addLayout(controls)

        # Custom Tab Widget with detach feature
        self.tabs = DraggableTabWidget()
        layout.addWidget(self.tabs)

        self.setCentralWidget(container)

    def add_empty_tab(self):
        placeholder = QLabel("Empty Tab – Add a Plot Later")
        placeholder.setAlignment(Qt.AlignCenter)
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        tab_layout.addWidget(placeholder)
        self.tabs.addTab(tab, "Empty")

    def add_plot_tab(self):
        plot_kind = self.plot_type.currentText()
        x_col = self.x_select.currentText()
        y_col = self.y_select.currentText()
        x_label = self.x_label_input.text() or x_col
        y_label = self.y_label_input.text() or y_col
        title = self.title_input.text() or f"{plot_kind}: {x_col} vs {y_col}"

        fig, ax = plt.subplots(figsize=(7, 5))

        try:
            if plot_kind == "Histogram":
                self.df[x_col].plot(kind="hist", bins=20, edgecolor="black", ax=ax)
            elif plot_kind == "Boxplot":
                self.df[[x_col]].plot(kind="box", ax=ax)
            elif plot_kind == "Scatter" and y_col in self.df.columns:
                self.df.plot(kind="scatter", x=x_col, y=y_col, ax=ax)
            elif plot_kind == "Heatmap":
                sns.heatmap(self.df.select_dtypes("number").corr(), annot=True, cmap="coolwarm", ax=ax)
            elif plot_kind == "KDE":
                sns.kdeplot(self.df[x_col], fill=True, ax=ax)
            elif plot_kind == "Violin":
                sns.violinplot(data=self.df[[x_col]], ax=ax)
            elif plot_kind == "Time Series":
                cols = [c for c in [x_col, y_col] if c in self.df.columns]
                self.df[cols].plot(ax=ax, legend=True)

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(title)

        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center")

        canvas = FigureCanvas(fig)

        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        tab_layout.addWidget(canvas)

        save_btn = QPushButton("Save Plot")
        save_btn.clicked.connect(lambda: self._save_plot(canvas))
        tab_layout.addWidget(save_btn)

        self.tabs.addTab(tab, title)

    def _save_plot(self, canvas):
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Plot", "", "PNG Image (*.png);;JPEG Image (*.jpg);;PDF File (*.pdf)"
        )
        if file_name:
            canvas.figure.savefig(file_name, bbox_inches="tight")


class VisualizeWindow(QMainWindow):
    def __init__(self, df=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Visualize Data")
        self.setGeometry(240, 240, 900, 600)

        self.panel = VisualizeDialog(df, self)
        self.setCentralWidget(self.panel)


class DashboardWindow(QMainWindow):
    def __init__(self, df=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Dashboard")
        self.setGeometry(260, 260, 1000, 700)

        self.panel = DashboardDialog(df, self)
        self.setCentralWidget(self.panel)


# —————————————————————————————————
#  MAIN APPLICATION WINDOW
# —————————————————————————————————

class DataCleaningApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.settings = QSettings("MyCompany", "DataCleanerApp")
        self.setWindowTitle("Dataspec")
        self.setGeometry(100, 100, 1150, 750)

        # Enable drag‐&‐drop on the main window
        self.setAcceptDrops(True)

        self.is_dark_mode = True
        self._undo_stack = []
        self.workflow_steps = []

        self.chunks = []
        self.current_chunk_idx = 0
        self.chunk_mode = False

        # Use a QStackedWidget to switch between “Welcome” and “Data” pages
        self.stacked = QStackedWidget()
        self.setCentralWidget(self.stacked)
        self.stacked.setContentsMargins(0, 0, 0, 0)
        if self.stacked.layout() is not None:
            self.stacked.layout().setSpacing(0)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Start with an empty DataFrame
        self.df = pd.DataFrame()
        self.model = PandasModel(self.df, workflow_callback=self.addWorkflowStep)
        self.model.data_changed.connect(self.updateSummary)
        self.model.cell_edited.connect(self.recordEdit)

        # Build pages & dock
        self.buildWelcomePage()
        self.buildDataPage()
        self.buildSummaryDock()
        self.buildTerminalDock()
        self.buildAIDock()
        # Build toolbar + icons
        self.initUI()

        # Apply initial stylesheet & transparency settings
        self.applyStyle()
        self.loadSettings()

    def openFitDialog(self):
        df = self.model.getDataFrame()
        if df.empty:
            QMessageBox.warning(self, "No Data", "Please load a dataset before fitting.")
            return
        dlg = FitDialog(df, self)
        dlg.exec_()


    def openFitWindow(self):
        df = self.model.getDataFrame()
        if df.empty:
            QMessageBox.warning(self, "No Data", "Please load a dataset before fitting.")
            return
        self.fit_window = FitWindow(df, self)
        self.fit_window.show()

    def openVisualizeWindow(self):
        df = self.model.getDataFrame()
        if df.empty:
            QMessageBox.warning(self, "No Data", "Please load a dataset before visualizing.")
            return
        self.viz_window = VisualizeWindow(df, self)
        self.viz_window.show()

    def openDashboardWindow(self):
        df = self.model.getDataFrame()
        if df.empty:
            QMessageBox.warning(self, "No Data", "Please load a dataset before dashboard view.")
            return
        self.dash_window = DashboardWindow(df, self)
        self.dash_window.show()

    def openTerminalWindow(self):
        self.term_window = TerminalWindow(self.model, self)
        self.term_window.show()

    def openAIWindow(self):
        self.ai_window = AIWindow(self.model, self)
        self.ai_window.show()

    def openExploreWindow(self):
        df = self.model.getDataFrame()
        if df.empty:
            QMessageBox.warning(self, "No Data", "Please load a dataset before exploring.")
            return

        # Keep reference so window isn’t garbage-collected
        self.explore_window = ExploreWindow(df, self)
        self.explore_window.show()

    def openAssistantWindow(self):
        self.assistant_window = AssistantWindow(self.model, self)
        self.assistant_window.show()

    def getIcon(self, name):
        """
        Return the correct icon (dark vs. light) from resources.
        """
        prefix = "icons_dark" if self.is_dark_mode else "icons_light"
        return QIcon(f":/resources/{prefix}/{name}.png")

    def initUI(self):
        toolbar = QToolBar("Main Toolbar")
        toolbar.setObjectName("mainToolbar")
        toolbar.setIconSize(QSize(20, 20))
        toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        toolbar.setMovable(False)
        toolbar.setContentsMargins(0, 0, 0, 0)

        # —— 1) Load File
        load_action = QAction(self.getIcon("load_file"), "Load File", self)
        load_action.setShortcut("Ctrl+O")
        load_action.setToolTip("Load CSV, TXT, or DAT (Ctrl+O)")
        load_action.triggered.connect(self.load_file)
        toolbar.addAction(load_action)

        # —— 2) Save File
        save_action = QAction(self.getIcon("save_file"), "Save File", self)
        save_action.setShortcut("Ctrl+S")
        save_action.setToolTip("Save cleaned data (Ctrl+S)")
        save_action.triggered.connect(self.save_file)
        toolbar.addAction(save_action)

        # —— 3) Drop NA Rows
        drop_na_action = QAction(self.getIcon("drop_na"), "Drop NA Rows", self)
        drop_na_action.setToolTip("Drop rows with missing values")
        drop_na_action.triggered.connect(self.drop_na_rows)
        toolbar.addAction(drop_na_action)

        # —— 4) Fill NA
        fill_na_action = QAction(self.getIcon("fill_na"), "Fill NA", self)
        fill_na_action.setToolTip("Fill missing values")
        fill_na_action.triggered.connect(self.openFillNADialog)
        toolbar.addAction(fill_na_action)

        # —— 5) Remove Selected
        remove_action = QAction(self.getIcon("remove_selected"), "Remove Selected", self)
        remove_action.setToolTip("Remove selected rows")
        remove_action.triggered.connect(self.remove_selected_rows)
        toolbar.addAction(remove_action)

        # —— 6) Rename Column
        rename_action = QAction(self.getIcon("rename_column"), "Rename Column", self)
        rename_action.setToolTip("Rename a column")
        rename_action.triggered.connect(self.renameColumnDialog)
        toolbar.addAction(rename_action)

        # —— 7) Delete Column
        delete_action = QAction(self.getIcon("delete_column"), "Delete Column", self)
        delete_action.setToolTip("Delete a column")
        delete_action.triggered.connect(self.deleteColumnDialog)
        toolbar.addAction(delete_action)

        # —— 8) Conditional Format
        cond_action = QAction(self.getIcon("conditional_format"), "Conditional Format", self)
        cond_action.setToolTip("Add conditional formatting rule")
        cond_action.triggered.connect(self.openConditionalDialog)
        toolbar.addAction(cond_action)

        # —— 9) Fit Data 
        fit_action = QAction(self.getIcon("fit"), "Fit Data", self)
        fit_action.setToolTip("Open Data Fitting panel")
        fit_action.triggered.connect(self.openFitWindow)
        toolbar.addAction(fit_action)

        # —— 10) Visualize
        
        # —— 10) Explore
        explore_action = QAction(self.getIcon("explore"), "Explore", self)
        explore_action.setToolTip("Explore data with histograms, scatter plots, heatmaps, and more")
        explore_action.triggered.connect(self.openExploreWindow)   # <-- Must point to the right function
        toolbar.addAction(explore_action)

        # Keep icon swapping consistent

        #self._icon_actions.append(explore_action)
        #self._icon_names.append("explore")

        # —— 12) Profile Report
        profile_action = QAction(self.getIcon("profile_report"), "Profile Report", self)
        profile_action.setToolTip("Generate an HTML profile report")
        profile_action.triggered.connect(self.generateProfileReport)
        toolbar.addAction(profile_action)

        # —— 13) Save Workflow
        save_work_action = QAction(self.getIcon("save_workflow"), "Save Workflow", self)
        save_work_action.setToolTip("Export cleaning steps as .py")
        save_work_action.triggered.connect(self.saveWorkflow)
        toolbar.addAction(save_work_action)

        # —— 14) Load Workflow
        load_work_action = QAction(self.getIcon("load_workflow"), "Load Workflow", self)
        load_work_action.setToolTip("Import a cleaning script")
        load_work_action.triggered.connect(self.loadWorkflow)
        toolbar.addAction(load_work_action)

        # —— 15) Undo
        undo_action = QAction(self.getIcon("undo"), "Undo", self)
        undo_action.setShortcut("Ctrl+Z")
        undo_action.setToolTip("Undo last action (Ctrl+Z)")
        undo_action.triggered.connect(self.undo)
        toolbar.addAction(undo_action)

        # —— 16) Redo
        #redo_action = QAction(self.getIcon("redo"), "Redo", self)
        #redo_action.setShortcut("Ctrl+Y")
        #redo_action.setToolTip("Redo last undone action (Ctrl+Y)")
        #redo_action.triggered.connect(self.redo)
        #toolbar.addAction(redo_action)

        # —— Terminal
        terminal_action = QAction(self.getIcon("terminal"), "Terminal", self)
        terminal_action.setToolTip("Open embedded Python terminal")
        terminal_action.triggered.connect(self.openTerminalWindow)
        toolbar.addAction(terminal_action)

        # —— AI Assistant
       # ai_action = QAction(self.getIcon("ai"), "AI Assistant", self)
       # ai_action.setToolTip("Open AI Assistant tab (offline GPT4All)")
       # ai_action.triggered.connect(self.openAIWindow)
       # toolbar.addAction(ai_action)

        assistant_action = QAction(self.getIcon("assistant_action"), "Assistant", self)
        assistant_action.setToolTip("Ask about Dataspec features or your dataset")
        assistant_action.triggered.connect(self.openAssistantWindow)
        toolbar.addAction(assistant_action)

        # Dataspec Assistant
        #assistant_action = QAction(self.getIcon("assistant"), "Dataspec Assistant", self)
        #assistant_action.setShortcut("Ctrl+Shift+A")
        #assistant_action.setToolTip("Ask about toolbar buttons, terminal, workflows (Ctrl+Shift+A)")
        #assistant_action.triggered.connect(lambda: self.assistant_dock.show())
        #toolbar.addAction(assistant_action)


        # —— 17) Settings
        settings_action = QAction(self.getIcon("setting"), "Settings", self)
        settings_action.setToolTip("Open Settings / Help")
        settings_action.triggered.connect(self.openSettingsDialog)
        toolbar.addAction(settings_action)

        # —— 18) Dark / Light Toggle
        initial_text = "Light Mode" if self.is_dark_mode else "Dark Mode"
        self.dark_mode_action = QAction(self.getIcon("toggle"), initial_text, self)
        self.dark_mode_action.setCheckable(True)
        self.dark_mode_action.setChecked(self.is_dark_mode)
        self.dark_mode_action.triggered.connect(self.toggleDarkLight)
        toolbar.addAction(self.dark_mode_action)

        # —— 19) Home
        home_action = QAction(self.getIcon("home"), "Home", self)
        home_action.setToolTip("Home (Welcome Screen)")
        home_action.triggered.connect(lambda: self.stacked.setCurrentIndex(0))
        toolbar.addAction(home_action)

        # ——————————————————————————————————————————————
        #  Now define your icon‐swap lists *once*
        # ——————————————————————————————————————————————
        self._icon_actions = [
            load_action, save_action, drop_na_action, fill_na_action,
            remove_action, rename_action, delete_action, cond_action,
            fit_action, explore_action, profile_action,
            save_work_action, load_work_action, undo_action, terminal_action,
            settings_action, self.dark_mode_action, home_action
        ]

        self._icon_names = [
            "load_file", "save_file", "drop_na", "fill_na",
            "remove_selected", "rename_column", "delete_column", "conditional_format",
            "fit", "explore", "profile_report",
            "save_workflow", "load_workflow", "undo", "terminal",
            "setting", "toggle", "home"
        ]

        # ——————————————————————————————————————————————
        #  Finally: spacer + add toolbar to window
        # ——————————————————————————————————————————————
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(spacer)

        self.addToolBar(toolbar)

    def applyStyle(self):
        """
        Apply the current QSS (dark or light), force the stacked widget to be transparent
        (so our WelcomeWidget paintEvent is responsible), and swap every toolbar icon.
        """
        QApplication.setStyle("Fusion")
        if self.is_dark_mode:
            self.setStyleSheet(DARK_QSS)
        else:
            self.setStyleSheet(LIGHT_QSS)

        # Force the central stacked widget background to be transparent
        self.stacked.setStyleSheet("background: transparent;")

        # Update each toolbar icon
        for action, name in zip(self._icon_actions, self._icon_names):
            action.setIcon(self.getIcon(name))

        # Toggle button’s text is always name of the *other* mode
        if self.is_dark_mode:
            self.dark_mode_action.setText("Light Mode")
        else:
            self.dark_mode_action.setText("Dark Mode")

        # If currently on Welcome page, update its logo/text colors
        if self.stacked.currentIndex() == 0:
            w = self.stacked.widget(0)
            if isinstance(w, WelcomeWidget):
                w.updateLogoAndTextColors(self.is_dark_mode)

        self.repaint()


    def toggleDarkLight(self, checked):
        """
        Flip is_dark_mode, reapply QSS & transparency.
        """
        self.is_dark_mode = bool(checked)
        self.applyStyle()


    def loadSettings(self):
        geometry = self.settings.value("mainWindowGeometry", None)
        if isinstance(geometry, QByteArray):
            self.restoreGeometry(geometry)
        state = self.settings.value("mainWindowState", None)
        if isinstance(state, QByteArray):
            self.restoreState(state)

        # Only show after we've applied QSS
        self.show()


    def closeEvent(self, event):
        self.settings.setValue("mainWindowGeometry", self.saveGeometry())
        self.settings.setValue("mainWindowState", self.saveState())
        super().closeEvent(event)


    def buildWelcomePage(self):
        # 1) Create the WelcomeWidget and add it to the stacked widget
        welcome_widget = WelcomeWidget(self)
        welcome_widget.setObjectName("welcomePage")
        welcome_widget.setContentsMargins(0, 0, 0, 0)
        self.stacked.addWidget(welcome_widget)

    def buildTerminalDock(self):
    #"""Adds the dockable Terminal to the main window."""
        self.terminal_dock = TerminalDock(self, self.model)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.terminal_dock)
        self.terminal_dock.hide()  # hidden until user opens it

        # 2) No additional margin tweak needed here now.

    
    def buildAIDock(self):
        self.ai_dock = AITabDock(self, self.model)
        self.addDockWidget(Qt.RightDockWidgetArea, self.ai_dock)
        self.ai_dock.hide()


    def buildDataPage(self):
        data_widget = QWidget()
        data_widget.setObjectName("centralWidget")
        data_layout = QVBoxLayout(data_widget)
        data_layout.setContentsMargins(12, 8, 12, 8)
        data_layout.setSpacing(6)

        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Type to filter and press Enter…")
        self.filter_input.returnPressed.connect(self.applyFilter)
        data_layout.addWidget(self.filter_input)

        self.table_view = QTableView()
        self.table_view.setModel(self.model)
        self.table_view.setSelectionBehavior(QTableView.SelectRows)
        self.table_view.setSelectionMode(QTableView.ExtendedSelection)
        self.table_view.setSortingEnabled(True)
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table_view.customContextMenuRequested.connect(self.openContextMenu)
        data_layout.addWidget(self.table_view)

        # ─────── New: Interactive Plot Section Below the Table ───────
        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(6)

        # Combo boxes to choose X and Y column for the interactive plot
        self.plot_x_combo = QComboBox()
        self.plot_y_combo = QComboBox()
        self.plot_x_combo.setToolTip("Select X‐axis column")
        self.plot_y_combo.setToolTip("Select Y‐axis column")
        self.plot_x_combo.currentTextChanged.connect(self._on_column_change)
        self.plot_y_combo.currentTextChanged.connect(self._on_column_change)

        combo_layout = QFormLayout()
        combo_layout.addRow("X‐Axis:", self.plot_x_combo)
        combo_layout.addRow("Y‐Axis:", self.plot_y_combo)
        plot_layout.addLayout(combo_layout)

        # Matplotlib canvas for interactive plot
        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.canvas.figure.subplots()
        plot_layout.addWidget(self.canvas)

        # Controls below the plot for color, style, best‐fit, etc.
        controls_layout = QHBoxLayout()

        # Color selector for scatter points
        self.color_button_plot = QPushButton("Point Color")
        self.color_button_plot.clicked.connect(self._choose_scatter_color)
        controls_layout.addWidget(self.color_button_plot)

        # Checkbox: Show best‐fit line
        self.best_fit_button = QPushButton("Toggle Best‐Fit")
        self.best_fit_button.setCheckable(True)
        self.best_fit_button.toggled.connect(self._on_column_change)
        controls_layout.addWidget(self.best_fit_button)

        # Download plot button (prominent)
        self.download_plot_button = QPushButton("Download Plot")
        self.download_plot_button.setStyleSheet("""
            QPushButton {
                border: 2px solid #008CBA;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #008CBA;
                color: #FFFFFF;
            }
        """)
        self.download_plot_button.clicked.connect(self._download_plot)
        controls_layout.addWidget(self.download_plot_button)

        plot_layout.addLayout(controls_layout)
        data_layout.addWidget(plot_container)
        # ────────────────────────────────────────────────────────────────

        self.stacked.addWidget(data_widget)


    def buildSummaryDock(self):
        self.summary_dock = QDockWidget("Data Summary", self)
        self.summary_dock.setObjectName("summaryDock")
        self.summary_dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self.summary_dock.setFeatures(QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable)
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_dock.setWidget(self.summary_text)
        self.addDockWidget(Qt.RightDockWidgetArea, self.summary_dock)
        self.summary_dock.hide()


    # ——————————————————————
    #  Toolbar Button Slots
    # ——————————————————————

    def load_file(self, file_name=None):
        """
        If file_name is provided (from drag‐&‐drop), use it.
        Otherwise, open QFileDialog.
        Supports CSV, TXT, DAT.
        """
        if not file_name:
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "Open CSV, TXT, or DAT",
                "",
                "CSV Files (*.csv);;Text Files (*.txt);;DAT Files (*.dat);;All Files (*)"
            )
        if not file_name:
            return

        try:
            ext = os.path.splitext(file_name)[1].lower()
            if ext not in [".csv", ".txt", ".dat"]:
                QMessageBox.warning(self, "Unsupported Format", "Only CSV, TXT, or DAT are supported.")
                return

             # choose parser based on extension
            if ext == ".csv":
                df = pd.read_csv(file_name)
            else:  # .txt or .dat
                df = pd.read_csv(file_name, sep=None, engine="python")

            file_size = os.path.getsize(file_name)
            threshold = 10 * 1024 * 1024  # 10 MB
            if file_size > threshold:
                size, ok = QInputDialog.getInt(
                    self,
                    "Chunked Loading",
                    "Rows per chunk (file > 10MB)?",
                    50000,
                    1000,
                    1000000,
                    1000
                )
                if not ok:
                    df = pd.read_csv(file_name)
                    self.chunk_mode = False
                else:
                    self.chunk_mode = True
                    self.chunks = []
                    total_rows = sum(1 for _ in open(file_name, "r")) - 1
                    total_chunks = (total_rows // size) + 1
                    progress = QProgressDialog("Reading file in chunks…", "Cancel", 0, total_chunks, self)
                    progress.setWindowModality(Qt.WindowModal)
                    progress.setAutoClose(True)
                    progress.setValue(0)

                    reader = pd.read_csv(file_name, chunksize=size)
                    for i, chunk in enumerate(reader):
                        QApplication.processEvents()
                        if progress.wasCanceled():
                            break
                        self.chunks.append(chunk)
                        progress.setValue(i + 1)
                    progress.close()

                    if not self.chunks:
                        QMessageBox.warning(self, "Warning", "No data loaded from chunks.")
                        return
                    self.current_chunk_idx = 0
                    df = self.chunks[self.current_chunk_idx]
                    self.status_bar.showMessage(
                        f"Loaded chunk 1 of {len(self.chunks)}", 4000
                    )
            else:
                self.chunk_mode = False
                df = pd.read_csv(file_name)

            self.pushUndoState()
            self.df = df.copy()
            self.model.update_dataframe(df)
            safe_path = file_name.replace("'''", "\\'\\'\\'")
            self.workflow_steps = [f"df = pd.read_csv(r'''{safe_path}''')"]
            self.status_bar.showMessage(
                f"Loaded: {os.path.basename(file_name)}   |   {df.shape[0]}×{df.shape[1]}", 5000
            )

            # When data is loaded, populate the X/Y plot combos:
            numeric_cols = self.df.select_dtypes(include="number").columns.tolist()
            self.plot_x_combo.clear()
            self.plot_y_combo.clear()
            self.plot_x_combo.addItems(numeric_cols)
            self.plot_y_combo.addItems(numeric_cols)

            # Switch to data page
            self.stacked.setCurrentIndex(1)
            self.summary_dock.show()
            self.updateSummary()

            # Immediately redraw the plot if both combos have something
            if len(numeric_cols) >= 2:
                self._on_column_change()

        except Exception as e:
            QMessageBox.critical(self, "Error Loading File", str(e))


    def save_file(self):
        if self.model.rowCount() == 0:
            QMessageBox.warning(self, "Warning", "No data to save.")
            return
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Cleaned File", "", "CSV Files (*.csv);;Excel Files (*.xlsx);;JSON Files (*.json)"
        )
        if not file_name:
            return
        try:
            df_to_save = self.model.getDataFrame()
            lower = file_name.lower()
            if lower.endswith(".csv"):
                df_to_save.to_csv(file_name, index=False)
            elif lower.endswith(".xlsx"):
                df_to_save.to_excel(file_name, index=False)
            elif lower.endswith(".json"):
                df_to_save.to_json(file_name, orient="records", lines=True)
            else:
                df_to_save.to_csv(file_name + ".csv", index=False)
            self.status_bar.showMessage(f"Saved: {os.path.basename(file_name)}", 4000)
        except Exception as e:
            QMessageBox.critical(self, "Error Saving File", str(e))


    def drop_na_rows(self):
        if self.model.rowCount() == 0:
            QMessageBox.warning(self, "Warning", "No data loaded.")
            return
        self.pushUndoState()
        self.model.dropAllNARows()
        self.addWorkflowStep("df = df.dropna()")
        self.status_bar.showMessage("Dropped all rows with NA", 4000)
        self.updateSummary()


    def openFillNADialog(self):
        if self.model.rowCount() == 0:
            QMessageBox.warning(self, "Warning", "No data loaded.")
            return
        dialog = FillNADialog(self)
        if dialog.exec_() == QDialog.Accepted:
            method, const = dialog.getValues()
            self.pushUndoState()
            if method == "Constant" and const == "":
                QMessageBox.warning(self, "Warning", "Please enter a constant.")
                return
            const_value = None
            if method == "Constant":
                try:
                    const_value = float(const)
                except ValueError:
                    const_value = const
            self.model.fillNARows(method, const_value)
            if method == "Constant":
                self.addWorkflowStep(f"df = df.fillna({repr(const_value)})")
            else:
                m = method.lower().replace(" ", "_")
                self.addWorkflowStep(f"df = df.fillna(method='{m}')")
            self.status_bar.showMessage(f"Filled NA using {method}", 4000)
            self.updateSummary()


    def remove_selected_rows(self):
        if self.model.rowCount() == 0:
            QMessageBox.warning(self, "Warning", "No data loaded.")
            return
        selected = self.table_view.selectionModel().selectedRows()
        if not selected:
            QMessageBox.information(self, "Info", "No rows selected.")
            return
        row_indices = sorted(set(idx.row() for idx in selected), reverse=True)
        self.pushUndoState()
        self.model.removeRows(row_indices)
        self.addWorkflowStep(f"df = df.drop(df.index[{row_indices}]).reset_index(drop=True)")
        self.status_bar.showMessage(f"Removed {len(row_indices)} selected row(s)", 4000)
        self.updateSummary()


    def renameColumnDialog(self):
        if self.model.columnCount() == 0:
            QMessageBox.warning(self, "Warning", "No columns to rename.")
            return
        items = list(self.model.getDataFrame().columns)
        old_name, ok = QInputDialog.getItem(self, "Select Column", "Column to Rename:", items, 0, False)
        if ok and old_name:
            new_name, ok2 = QInputDialog.getText(self, "New Column Name", f"Rename '{old_name}' to:")
            if ok2 and new_name:
                self.pushUndoState()
                self.model.renameColumn(old_name, new_name)
                self.addWorkflowStep(f"df = df.rename(columns={{'{old_name}':'{new_name}'}})")
                self.status_bar.showMessage(f"Renamed column '{old_name}' to '{new_name}'", 4000)
                self.updateSummary()


    def deleteColumnDialog(self):
        if self.model.columnCount() == 0:
            QMessageBox.warning(self, "Warning", "No columns to delete.")
            return
        items = list(self.model.getDataFrame().columns)
        col_name, ok = QInputDialog.getItem(self, "Select Column", "Column to Delete:", items, 0, False)
        if ok and col_name:
            idx = items.index(col_name)
            confirm = QMessageBox.question(self, "Confirm Delete", f"Delete column '{col_name}'?",
                                           QMessageBox.Yes | QMessageBox.No)
            if confirm == QMessageBox.Yes:
                self.pushUndoState()
                self.model.deleteColumn(idx)
                self.addWorkflowStep(f"df = df.drop(columns=['{col_name}'])")
                self.status_bar.showMessage(f"Deleted column '{col_name}'", 4000)
                self.updateSummary()


    def openConditionalDialog(self):
        if self.model.columnCount() == 0:
            QMessageBox.warning(self, "Warning", "No columns available.")
            return
        columns = list(self.model.getDataFrame().columns)
        dialog = ConditionalDialog(columns, self)
        if dialog.exec_() == QDialog.Accepted:
            col, op, val, color = dialog.getValues()
            self.model.addConditionalRule(col, op, val, color)
            self.addWorkflowStep(f"# Conditional rule: {col} {op} {val} → color {color}")
            self.status_bar.showMessage(f"Added conditional rule on '{col}'", 4000)


    def openVisualizeDialog(self):
        df = self.model.getDataFrame()
        if df.empty:
            QMessageBox.warning(self, "Warning", "No data to visualize.")
            return
        dialog = VisualizeDialog(df, self)
        dialog.exec_()


    def openDashboardDialog(self):
        df = self.model.getDataFrame()
        if df.empty:
            QMessageBox.warning(self, "Warning", "No data for dashboard.")
            return
        dialog = DashboardDialog(df, self)
        dialog.exec_()


    def generateProfileReport(self):
        if ProfileReport is None:
            QMessageBox.warning(
                self,
                "Dependency Missing",
                "ydata-profiling is not installed or incompatible.\n"
                "Run: pip install ydata-profiling\n"
                "Or create a Python 3.10 environment and:\n"
                "  conda install -c conda-forge pandas-profiling numba=0.55.2"
            )
            return
        df = self.model.getDataFrame()
        if df.empty:
            QMessageBox.warning(self, "Warning", "No data to profile.")
            return
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Profile Report", "", "HTML Files (*.html);;All Files (*)"
        )
        if not file_name:
            return
        progress = QProgressDialog("Generating profile report…", None, 0, 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(200)
        progress.show()
        try:
            report = ProfileReport(df, minimal=True)
            report.to_file(file_name)
        except Exception as e:
            QMessageBox.critical(self, "Error Generating Profile", str(e))
            progress.close()
            return
        progress.close()
        QMessageBox.information(self, "Profile Report Saved", f"Report saved to:\n{file_name}")


    def saveWorkflow(self):
        if not self.workflow_steps:
            QMessageBox.information(self, "Info", "No actions recorded.")
            return
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Workflow", "", "Python Files (*.py);;All Files (*)"
        )
        if not file_name:
            return
        try:
            with open(file_name, "w") as f:
                f.write("import pandas as pd\n")
                f.write("# Load data before running these steps: df = pd.read_csv('your_file.csv')\n\n")
                for step in self.workflow_steps:
                    f.write(step + "\n")
            self.status_bar.showMessage(f"Workflow saved: {os.path.basename(file_name)}", 4000)
        except Exception as e:
            QMessageBox.critical(self, "Error Saving Workflow", str(e))


    def loadWorkflow(self):
        if self.model.getDataFrame().empty:
            QMessageBox.warning(self, "Warning", "Load a dataset first before applying a workflow.")
            return
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Load Workflow", "", "Python Files (*.py);;All Files (*)"
        )
        if not file_name:
            return
        try:
            with open(file_name, "r") as f:
                code = f.read()
            local_vars = {"pd": pd, "df": self.model.getDataFrame()}
            exec(code, {}, local_vars)
            new_df = local_vars.get("df", None)
            if isinstance(new_df, pd.DataFrame):
                self.pushUndoState()
                self.model.update_dataframe(new_df)
                self.workflow_steps = code.splitlines()[2:]
                self.status_bar.showMessage(f"Workflow applied: {os.path.basename(file_name)}", 4000)
                self.updateSummary()
            else:
                QMessageBox.warning(self, "Warning", "Workflow did not produce a DataFrame named 'df'.")
        except Exception as e:
            QMessageBox.critical(self, "Error Loading Workflow", str(e))


    def pushUndoState(self):
        df_copy = self.model.getDataFrame()
        self._undo_stack.append(df_copy)
        if len(self._undo_stack) > 10:
            self._undo_stack.pop(0)


    def undo(self):
        if not self._undo_stack:
            QMessageBox.information(self, "Info", "Nothing to undo.")
            return
        last_df = self._undo_stack.pop()
        self.model.update_dataframe(last_df)
        self.status_bar.showMessage("Undo completed", 4000)
        self.updateSummary()


    def redo(self):
        # You can implement a redo stack similarly to undo if desired.
        QMessageBox.information(self, "Info", "Redo is not yet implemented.")
        return


    def addWorkflowStep(self, step):
        self.workflow_steps.append(step)


    def recordEdit(self, row, col, new_val):
        col_name = self.model.getDataFrame().columns[col]
        if isinstance(new_val, str):
            new_val_repr = repr(new_val)
        else:
            new_val_repr = str(new_val)
        self.addWorkflowStep(f"df.iat[{row}, {col}] = {new_val_repr}")


    def openContextMenu(self, position):
        index = self.table_view.indexAt(position)
        menu = QMenu()

        if index.isValid():
            delete_row_action = QAction("Delete This Row", self)
            delete_row_action.setToolTip("Delete the clicked row")
            delete_row_action.triggered.connect(lambda: self.deleteRow(index.row()))
            menu.addAction(delete_row_action)

        delete_col_action = QAction("Delete This Column", self)
        delete_col_action.setToolTip("Delete the clicked column")
        delete_col_action.triggered.connect(lambda: self.deleteColumn(index.column()))
        menu.addAction(delete_col_action)

        rename_col_action = QAction("Rename This Column", self)
        rename_col_action.setToolTip("Rename the clicked column")
        rename_col_action.triggered.connect(lambda: self.renameColumnSpecific(index.column()))
        menu.addAction(rename_col_action)

        menu.exec_(self.table_view.viewport().mapToGlobal(position))


    def deleteRow(self, row):
        self.pushUndoState()
        self.model.removeRows([row])
        self.addWorkflowStep(f"df = df.drop(index=[{row}]).reset_index(drop=True)")
        self.status_bar.showMessage(f"Deleted row {row}", 4000)
        self.updateSummary()


    def renameColumnSpecific(self, col_index):
        old_name = self.model.getDataFrame().columns[col_index]
        new_name, ok = QInputDialog.getText(self, "New Column Name", f"Rename '{old_name}' to:")
        if ok and new_name:
            self.pushUndoState()
            self.model.renameColumn(old_name, new_name)
            self.addWorkflowStep(f"df = df.rename(columns={{'{old_name}':'{new_name}'}})")
            self.status_bar.showMessage(f"Renamed column '{old_name}' to '{new_name}'", 4000)
            self.updateSummary()


    def deleteColumn(self, col_index):
        col_name = self.model.getDataFrame().columns[col_index]
        confirm = QMessageBox.question(self, "Confirm Delete", f"Delete column '{col_name}'?",
                                       QMessageBox.Yes | QMessageBox.No)
        if confirm == QMessageBox.Yes:
            self.pushUndoState()
            self.model.deleteColumn(col_index)
            self.addWorkflowStep(f"df = df.drop(columns=['{col_name}'])")
            self.status_bar.showMessage(f"Deleted column '{col_name}'", 4000)
            self.updateSummary()


    def applyFilter(self):
        text = self.filter_input.text()
        self.model.filter(text)
        self.addWorkflowStep(
            f"df = df[df.astype(str).apply(lambda row: row.str.contains({repr(text)}, case=False, na=False)).any(axis=1)].reset_index(drop=True)"
        )
        self.status_bar.showMessage(f"Filter applied: '{text}'   |   {self.model.rowCount()} rows", 4000)
        self.updateSummary()


    def updateSummary(self):
        df = self.model.getDataFrame()
        if df.empty:
            self.summary_text.setPlainText("No data loaded.")
            return
        lines = []
        lines.append(f"Rows: {df.shape[0]}")
        lines.append(f"Columns: {df.shape[1]}")
        lines.append("\nMissing Values per Column:")
        missing = df.isna().sum()
        for col, cnt in missing.items():
            lines.append(f"  {col}: {cnt}")
        lines.append("\nColumn Data Types:")
        dtypes = df.dtypes
        for col, dt in dtypes.items():
            lines.append(f"  {col}: {dt}")
        self.summary_text.setPlainText("\n".join(lines))


    # ——————————————————————————
    #  Interactive Plot Helpers
    # ——————————————————————————

    def _on_column_change(self):
        """
        Redraw the scatter plot (and best‐fit line if toggled) whenever X or Y combo changes.
        """
        x_col = self.plot_x_combo.currentText()
        y_col = self.plot_y_combo.currentText()
        show_best_fit = self.best_fit_button.isChecked()

        if not x_col or not y_col:
            return

        try:
            # Extract x and y arrays
            x = self.df[x_col].dropna().values
            y = self.df[y_col].dropna().values

            # We need to keep them aligned; easiest is to drop any rows where either is NA
            valid_mask = ~pd.isna(self.df[x_col]) & ~pd.isna(self.df[y_col])
            x = self.df.loc[valid_mask, x_col].values
            y = self.df.loc[valid_mask, y_col].values

            # Sort by x to draw lines properly
            idx_sort = x.argsort()
            x_sorted = x[idx_sort]
            y_sorted = y[idx_sort]

            self.ax.clear()
            self.ax.scatter(x_sorted, y_sorted, color="#008CBA", edgecolor="#E0E0E0", label="Data Points")
            self.ax.set_xlabel(x_col, color="#E0E0E0")
            self.ax.set_ylabel(y_col, color="#E0E0E0")
            self.ax.set_title(f"{y_col} vs {x_col}", color="#E0E0E0")
            self.ax.tick_params(colors="#E0E0E0")

            if show_best_fit and len(x_sorted) >= 2:
                # Compute linear best‐fit (Maple/NumPy polyfit of degree 1)
                coeffs = np.polyfit(x_sorted, y_sorted, deg=1)
                poly = np.poly1d(coeffs)
                y_fit = poly(x_sorted)
                self.ax.plot(x_sorted, y_fit, color="#FF4500", linestyle="--", linewidth=2, label="Linear Fit")

                # Compute R² if desired or show coefficients
                slope, intercept = coeffs
                r2 = np.corrcoef(y_sorted, y_fit)[0, 1]**2

                # Display best‐fit info above the plot
                info_text = f"Slope: {slope:.4f}   Intercept: {intercept:.4f}   R²: {r2:.4f}"
                self.ax.text(
                    0.02, 0.95, info_text,
                    transform=self.ax.transAxes,
                    color="#E0E0E0", fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='#333333', alpha=0.5)
                )

                # OPTIONAL: Add interpolation/extrapolation segments if you like
                # For brevity, we only show the linear best‐fit line here.

                # Enable hover for best‐fit line: use mplcursors if installed
                try:
                    import mplcursors
                except ImportError:
                    mplcursors = None
                    mplcursors.cursor(self.ax.lines[-1], hover=True).connect(
                        "add", lambda sel: sel.annotation.set_text(
                            f"x={sel.target[0]:.3f}\ny={sel.target[1]:.3f}"
                        )
                    )
                except ImportError:
                    pass

            # Legend
            self.ax.legend(facecolor="#141414", edgecolor="#E0E0E0", labelcolor="#E0E0E0")
            self.figure.tight_layout()
            self.canvas.draw()

        except Exception as e:
            # In case the arrays don’t align perfectly or other errors
            print("Error in _on_column_change:", e)


    def _choose_scatter_color(self):
        """
        Open QColorDialog so user can choose scatter point color.
        """
        color = QColorDialog.getColor()
        if color.isValid():
            # Update the button’s background so user sees chosen color
            self.color_button_plot.setStyleSheet(f"background-color: {color.name()}; color: #000000;")
            self.scatter_color = color.name()
            # Redraw the plot with the new scatter color
            self._on_column_change()


    def _download_plot(self):
        """
        Save the current canvas (PNG) and optionally the best‐fit info text to a file.
        """
        if self.figure is None:
            return
        fname, _ = QFileDialog.getSaveFileName(self, "Save Plot Image", "", "PNG Files (*.png);;All Files (*)")
        if not fname:
            return
        if not fname.lower().endswith(".png"):
            fname += ".png"
        try:
            self.figure.savefig(fname, dpi=300, facecolor=self.figure.get_facecolor())
            self.status_bar.showMessage(f"Plot saved as {os.path.basename(fname)}", 4000)
        except Exception as e:
            QMessageBox.critical(self, "Error Saving Plot", str(e))


    # ——————————————————————————
    #  Drag & Drop Events
    # ——————————————————————————

    def dragEnterEvent(self, event):
        """
        Accept drag if it is a file with extension .csv, .txt, or .dat
        """
        mime = event.mimeData()
        if mime.hasUrls():
            urls = mime.urls()
            if urls and urls[0].isLocalFile():
                ext = os.path.splitext(urls[0].toLocalFile())[1].lower()
                if ext in [".csv", ".txt", ".dat"]:
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event):
        """
        On drop, extract the first local file URL and call load_file() with it.
        """
        mime = event.mimeData()
        if mime.hasUrls():
            urls = mime.urls()
            if urls and urls[0].isLocalFile():
                file_path = urls[0].toLocalFile()
                self.load_file(file_path)
                event.acceptProposedAction()
                return
        event.ignore()


    # ——————————————————————————
    #  Settings / Help Dialog
    # ——————————————————————————

    def openSettingsDialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Settings & Help")
        dlg.setModal(True)
        dlg.resize(600, 500)
        dlg.setStyleSheet("""
            QDialog {
                background: #141414;
                color: #E0E0E0;
                border-radius: 8px;
            }
            QLabel, QTextEdit, QLineEdit, QComboBox {
                background: #0D0D0D;
                color: #E0E0E0;
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 6px 8px;
                font-size: 14px;
            }
            QPushButton {
                background: #008CBA;
                color: #FFFFFF;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 14px;
            }
            QPushButton:hover {
                background: #005F75;
            }
        """)
        layout = QVBoxLayout(dlg)

        # Tab widget: “Help” and “Shortcuts”
        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane { background: #0D0D0D; border: none; }
            QTabBar::tab {
                background: #141414; color: #E0E0E0; padding: 6px;
                border: 1px solid #333333; border-bottom: none;
                border-top-left-radius: 4px; border-top-right-radius: 4px;
                min-width: 100px;
            }
            QTabBar::tab:selected {
                background: #008CBA; color: #0D0D0D;
            }
        """)
        layout.addWidget(tabs)

        # ─── Help Tab ──────────────────────────────────────────────────────────
        help_tab = QWidget()
        help_layout = QVBoxLayout(help_tab)
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setStyleSheet("background: #0D0D0D; color: #E0E0E0;")
        help_content = """
<h2>Dataspec – Your Complete Data Cleaning, Analysis & Reporting Tool</h2>

<p><b>Overview:</b> Dataspec helps you load, clean, explore, analyze, and report on tabular data 
without leaving a single interface. It combines powerful data-science tools with an 
easy-to-use GUI and offline AI assistance.</p>

<hr>
<h3>✨ Core Features</h3>
<ul>
  <li><b>File Loading:</b> Drag & Drop CSV, TXT, or DAT files, or use <kbd>Ctrl+O</kbd>. 
      Large files (>10MB) are automatically offered in chunked loading mode.</li>
  <li><b>Data Cleaning:</b> 
    <ul>
      <li>Drop rows with missing values.</li>
      <li>Fill gaps (mean, median, forward/backward fill, or constant values).</li>
      <li>Rename or delete columns directly.</li>
      <li>Delete or remove specific rows.</li>
      <li>Apply <b>conditional formatting</b> rules to highlight values or outliers.</li>
    </ul>
  </li>
  <li><b>Data Analysis:</b>
    <ul>
      <li>Interactive scatter plots with color control, best-fit lines, slope/intercept, and R² display.</li>
      <li>Histograms, box plots, and scatter plots (via <i>Visualize</i> menu).</li>
      <li>Correlation heatmaps, boxplots, and pairwise scatter plots in the <i>Dashboard</i>.</li>
      <li>Advanced fitting: Linear, Polynomial, Exponential, Gaussian, Lorentzian, Voigt, Custom expressions, and Bayesian regression (if PyMC is installed).</li>
    </ul>
  </li>
  <li><b>AI Assistant:</b> 
    <ul>
      <li>Offline GPT4All integration (no internet needed).</li>
      <li>Ask natural-language questions about your dataset.</li>
      <li>Get instant insights and explanations of data summaries and distributions.</li>
    </ul>
  </li>
  <li><b>Terminal:</b>
    <ul>
      <li>Embedded Python REPL connected to your dataset (`df`, `pd`, `np`).</li>
      <li>Supports inline results, syntax coloring, and persistent history.</li>
      <li>Run custom code and immediately update the table and plots.</li>
    </ul>
  </li>
  <li><b>Reports & Workflow:</b>
    <ul>
      <li>Generate automated <b>HTML Profile Reports</b> (via <i>ydata-profiling</i>).</li>
      <li>Export all cleaning steps as a reproducible Python script.</li>
      <li>Reload saved workflows and apply them to new datasets.</li>
    </ul>
  </li>
</ul>

<hr>
<h3>⚙️ Toolbar Guide</h3>
<p>The top toolbar contains shortcuts for all actions. Hover over any icon to see a tooltip.</p>
<ul>
  <li><b>Load / Save File</b> – Import or export datasets.</li>
  <li><b>Drop NA / Fill NA</b> – Handle missing data.</li>
  <li><b>Remove Rows / Rename Column / Delete Column</b> – Edit your dataset structure.</li>
  <li><b>Conditional Format</b> – Add rules to highlight cells.</li>
  <li><b>Fit Data</b> – Advanced curve fitting tools.</li>
  <li><b>Visualize</b> – Quick histogram plots.</li>
  <li><b>Dashboard</b> – Multi-plot exploration panel.</li>
  <li><b>Profile Report</b> – Full dataset profiling (HTML export).</li>
  <li><b>Save/Load Workflow</b> – Script your cleaning pipeline.</li>
  <li><b>Undo</b> – Roll back the last action.</li>
  <li><b>Terminal</b> – Open interactive Python console.</li>
  <li><b>AI Assistant</b> – Ask AI questions about your data.</li>
  <li><b>Settings</b> – Open this help window.</li>
  <li><b>Dark/Light Toggle</b> – Switch themes.</li>
  <li><b>Home</b> – Return to the Welcome page.</li>
</ul>

<hr>
<h3>⌨️ Shortcuts</h3>
<ul>
  <li><kbd>Ctrl+O</kbd> – Open File</li>
  <li><kbd>Ctrl+S</kbd> – Save File</li>
  <li><kbd>Ctrl+Z</kbd> – Undo</li>
  <li><kbd>Ctrl+T</kbd> – Toggle Dark/Light Mode</li>
  <li>More customizable shortcuts can be set under the <b>Shortcuts</b> tab.</li>
</ul>

<hr>
<p><i>Tip:</i> For best results, start by cleaning your dataset, then explore patterns visually, 
use AI or Terminal for deeper insights, and finally export a report or workflow.</p>
"""
        help_text.setHtml(help_content)
        help_layout.addWidget(help_text)
        tabs.addTab(help_tab, "Help")

        # ─── Shortcuts Tab ─────────────────────────────────────────────────────────
        shortcuts_tab = QWidget()
        shortcuts_layout = QFormLayout(shortcuts_tab)

        # Example: Show current mapping and allow user to rebind
        # For brevity, we'll show only a few entries here; you can expand as needed.
        self.shortcut_map = {
    # File operations
    "Load File": "Ctrl+O",
    "Save File": "Ctrl+S",

    # Editing
    "Undo": "Ctrl+Z",
    "Redo": "Ctrl+Y",
    "Drop NA Rows": "Ctrl+D",
    "Fill NA": "Ctrl+F",
    "Remove Selected Rows": "Ctrl+R",
    "Rename Column": "Ctrl+Shift+R",
    "Delete Column": "Ctrl+Del",

    # Analysis & Visualization
    "Fit Data": "Ctrl+Shift+F",
    "Visualize": "Ctrl+Shift+V",
    "Dashboard": "Ctrl+Shift+D",

    # AI & Terminal
    "AI Assistant": "Ctrl+I",
    "Open Terminal": "Ctrl+`",

    # Reports & Workflows
    "Profile Report": "Ctrl+P",
    "Save Workflow": "Ctrl+Shift+S",
    "Load Workflow": "Ctrl+Shift+O",

    # Navigation & Settings
    "Home": "Ctrl+H",
    "Toggle Dark/Light": "Ctrl+T",
    "Open Settings": "Ctrl+,"
}
        self.shortcut_edits = {}
        for action_name, default_keys in self.shortcut_map.items():
            edit = QLineEdit(default_keys)
            edit.setReadOnly(True)
            edit.setPlaceholderText("Click to remap")
            edit.mousePressEvent = lambda ev, name=action_name: self._remap_shortcut(name)
            shortcuts_layout.addRow(f"{action_name}:", edit)
            self.shortcut_edits[action_name] = edit

        tabs.addTab(shortcuts_tab, "Shortcuts")

        # ───────────────────────────────────────────────────────────────────────────
        dlg.exec_()


    def _remap_shortcut(self, action_name):
        """
        Prompt user to press new key combination, then update mapping.
        """
        new_key, ok = QInputDialog.getText(self, "Remap Shortcut", f"Press new shortcut for '{action_name}':")
        if ok and new_key:
            # Here you would update the actual QAction’s shortcut binding
            # e.g., self.some_action.setShortcut(new_key)
            self.shortcut_map[action_name] = new_key
            self.shortcut_edits[action_name].setText(new_key)
            self.status_bar.showMessage(f"Shortcut for '{action_name}' set to {new_key}", 4000)


    # —————————————————————————————————
    #  Overridden drag‐drop events (already shown above)
    # —————————————————————————————————

    #  ... (dragEnterEvent & dropEvent already implemented above) ...


    # —————————————————————————————————
    #  Mac: NSOpenPanel warning is normal; ignore.
    # —————————————————————————————————


def main():
    app = QApplication(sys.argv)
    window = DataCleaningApp()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
