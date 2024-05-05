from Ui_illuMap import Ui_Form
from PyQt5.QtWidgets import QWidget, QHBoxLayout

from utli import ImgFigure


class IlluMapWindow(QWidget, Ui_Form):
    def __init__(self):
        super(IlluMapWindow, self).__init__()
        self.setupUi(self)

        self.figure = ImgFigure()
        self.layout = QHBoxLayout(self.groupBox)
        self.layout.addWidget(self.figure)

        self.colorMap = {'紅黃': 'OrRd_r',
                         '灰色': 'Greys',
                         '紅粉': 'RdPu',
                         '藍綠': 'GnBu',
                         '黃綠藍': 'YlGnBu',
                         '粉藍': 'PuBu'}


if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = IlluMapWindow()
    window.show()
    sys.exit(app.exec())
