from matplotlib.figure import Figure
from matplotlib.pyplot import imread
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from numpy import ndarray
from PyQt5.QtCore import QThread, pyqtSignal


class ImgFigure(FigureCanvas):
    def __init__(self, dpi=100):
        self.fig = Figure(dpi=dpi)
        super(ImgFigure, self).__init__(self.fig)
        self.axes = self.fig.add_subplot(111)
        self.axes.get_yaxis().set_visible(False)
        self.axes.get_xaxis().set_visible(False)


class WorkThread(QThread):

    finishSignal = pyqtSignal(ndarray, ndarray)

    def __init__(self, imgPath, progressBar, alpha, gamma):
        super(WorkThread, self).__init__()

        from LIME import LIME
        img = imread(imgPath)
        self.lime = LIME(img, alpha, gamma)
        self.lime.setMaximumSignal.connect(progressBar.setMaximum)
        self.lime.setValueSignal.connect(progressBar.setValue)

    def run(self):
        T = self.lime.optimizeIllumMap()
        R = self.lime.enhance()
        self.finishSignal.emit(T, R)

class HE(QThread):

    finishSignal = pyqtSignal(ndarray, ndarray)

    def __init__(self, imgPath, progressBar, alpha, gamma):
        super(HE, self).__init__()

        from LIME import LIME
        img = imread(imgPath)
        self.lime = LIME(img, alpha, gamma)
        self.lime.setMaximumSignal.connect(progressBar.setMaximum)
        self.lime.setValueSignal.connect(progressBar.setValue)

    def run(self):
        T = self.lime.optimizeIllumMap()
        R = self.lime.HE_enhance()
        self.finishSignal.emit(T, R)

class HSI(QThread):

    finishSignal = pyqtSignal(ndarray, ndarray)

    def __init__(self, imgPath, progressBar, alpha, gamma):
        super(HSI, self).__init__()

        from LIME import LIME
        img = imread(imgPath)
        self.lime = LIME(img, alpha, gamma)
        self.lime.setMaximumSignal.connect(progressBar.setMaximum)
        self.lime.setValueSignal.connect(progressBar.setValue)

    def run(self):
        T = self.lime.optimizeIllumMap()
        R = self.lime.HSI_enhance()
        self.finishSignal.emit(T, R)

class GC(QThread):

    finishSignal = pyqtSignal(ndarray, ndarray)

    def __init__(self, imgPath, progressBar, alpha, gamma):
        super(GC, self).__init__()

        from LIME import LIME
        img = imread(imgPath)
        self.lime = LIME(img, alpha, gamma)
        self.lime.setMaximumSignal.connect(progressBar.setMaximum)
        self.lime.setValueSignal.connect(progressBar.setValue)

    def run(self):
        T = self.lime.optimizeIllumMap()
        R = self.lime.GC_enhance()
        self.finishSignal.emit(T, R)

class CVC(QThread):

    finishSignal = pyqtSignal(ndarray, ndarray)

    def __init__(self, imgPath, progressBar, alpha, gamma):
        super(CVC, self).__init__()

        from LIME import LIME
        img = imread(imgPath)
        self.lime = LIME(img, alpha, gamma)
        self.lime.setMaximumSignal.connect(progressBar.setMaximum)
        self.lime.setValueSignal.connect(progressBar.setValue)

    def run(self):
        T = self.lime.optimizeIllumMap()
        R = self.lime.CVC_enhance()
        self.finishSignal.emit(T, R)