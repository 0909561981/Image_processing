import numpy as np
from scipy.fft import fft, ifft
from numpy.linalg import norm
from skimage import img_as_ubyte, img_as_float
from skimage.exposure import rescale_intensity
import cv2
from PyQt5.QtCore import QObject, pyqtSignal
from skimage.exposure import equalize_hist
from skimage.exposure import rescale_intensity
from skimage.color import rgb2gray

class LIME(QObject):

    setMaximumSignal = pyqtSignal(int)
    setValueSignal = pyqtSignal(int)

    def __init__(self, img, alpha=1, gamma=0.7, rho=2):
        super(LIME, self).__init__()
        self.L = img_as_float(img)
        self.row = self.L.shape[0]
        self.col = self.L.shape[1]

        self.alpha = alpha
        self.rho = rho
        self.gamma = gamma

        self.__toeplitzMatrix()
        self.__initIllumMap()

    def __initIllumMap(self):
        r = self.L[:, :, 0]
        g = self.L[:, :, 1]
        b = self.L[:, :, 2]
        self.T_hat = np.maximum(np.maximum(r, g), b)
        self.epsilon = norm(self.T_hat, ord='fro') * 0.001

        return self.T_hat

    def __toeplitzMatrix(self):

        def firstOrderDerivative(n, k=1):
            return (np.eye(n)) * (-1) + np.eye(n, k=k)

        self.dv = firstOrderDerivative(self.row)
        self.dh = firstOrderDerivative(self.col, -1)
        vecDD = np.zeros(self.row * self.col)
        vecDD[0] = 4
        vecDD[1] = -1
        vecDD[self.row] = -1
        vecDD[-1] = -1
        vecDD[-self.row] = -1
        self.vecDD = vecDD

    def __T_subproblem(self, G, Z, u):

        def vectorize(matrix):
            return matrix.T.ravel()

        def reshape(vector):
            return vector.reshape((self.row, self.col), order='F')

        X = G - Z / u
        Xv = X[:self.row, :]
        Xh = X[self.row:, :]
        temp = self.dv @ Xv + Xh @ self.dh
        numerator = fft(vectorize(2 * self.T_hat + u * temp))
        denominator = fft(self.vecDD * u) + 2
        T = ifft(numerator / denominator)
        T = np.real(reshape(T))
        return rescale_intensity(T, (0, 1), (0.0001, 1))

    def __derivative(self, matrix):
        v = self.dv @ matrix
        h = matrix @ self.dh
        return np.vstack([v, h])

    def __G_subproblem(self, T, Z, u, W):
        dT = self.__derivative(T)
        epsilon = self.alpha * W / u
        X = dT + Z / u
        return np.sign(X) * np.maximum(np.abs(X) - epsilon, 0)

    def __Z_subproblem(self, T, G, Z, u):
        dT = self.__derivative(T)
        return Z + u * (dT - G)

    def __u_subproblem(self, u):
        return u * self.rho

    def __weightingStrategy_1(self):
        self.W = np.ones((self.row * 2, self.col))

    def __weightingStrategy_2(self):
        dTv = self.dv @ self.T_hat
        dTh = self.T_hat @ self.dh
        Wv = 1 / (np.abs(dTv) + 1)
        Wh = 1 / (np.abs(dTh) + 1)
        self.W = np.vstack([Wv, Wh])

    def optimizeIllumMap(self):
        self.__weightingStrategy_2()

        T = np.zeros((self.row, self.col))
        G = np.zeros((self.row * 2, self.col))
        Z = np.zeros((self.row * 2, self.col))
        t = 0
        u = 1

        while True:
            T = self.__T_subproblem(G, Z, u)
            G = self.__G_subproblem(T, Z, u, self.W)
            Z = self.__Z_subproblem(T, G, Z, u)
            u = self.__u_subproblem(u)

            if t == 0:
                temp = norm((self.__derivative(T) - G), ord='fro')
                self.expert_t = np.ceil(2 * np.log(temp / self.epsilon))
                self.setMaximumSignal.emit(self.expert_t+1)

            t += 1
            self.setValueSignal.emit(t)

            if t >= self.expert_t:
                break

        self.T = T ** self.gamma
        return self.T

    def enhance(self):
        self.R = np.zeros(self.L.shape)
        for i in range(3):
            self.R[:, :, i] = self.L[:, :, i] / self.T
        self.R = rescale_intensity(self.R, (0, 1))
        self.R = img_as_ubyte(self.R)
        return self.R
    # 這裡沒有用到self.t，不會被參數影響...
    '''def HE_enhance(self):
        gray_image = rgb2gray(self.L)
        enhanced_gray_image = equalize_hist(gray_image)
        self.R = np.zeros(self.L.shape)
        for i in range(3):
            nonzero_mask = gray_image != 0
            self.R[:, :, i] = np.where(nonzero_mask, self.L[:, :, i] * enhanced_gray_image / np.where(nonzero_mask, gray_image, 1), self.L[:, :, i])
        self.R = rescale_intensity(self.R, (0, 1))
        self.R = img_as_ubyte(self.R)
        return self.R
    '''

    def HE_enhance(self):
        hsi_image = self.rgb_to_hsi(self.L)
        hsi_image = img_as_float(hsi_image)
        hsi_image[:, :, 2] = equalize_hist(hsi_image[:, :, 2])
        enhanced_image = self.hsi_to_rgb(hsi_image)
        enhanced_image = img_as_float(enhanced_image)
        self.R = rescale_intensity(enhanced_image, in_range=(0, 1), out_range=np.uint8)
        return self.R

    def rgb_to_hsi(self, image):
        with np.errstate(divide='ignore', invalid='ignore'):
            image = image.astype('float32') / 255.0
            r, g, b = cv2.split(image)
            
            i = (r + g + b) / 3.0

            min_rgb = np.minimum(np.minimum(r, g), b)
            s = 1 - 3 * min_rgb / (r + g + b + 1e-6)

            h = np.arccos(0.5 * ((r - g) + (r - b)) / (np.sqrt((r - g) ** 2 + (r - b) * (g - b)) + 1e-6))
            h[b > g] = 2 * np.pi - h[b > g]
            h = h / (2 * np.pi)

            return cv2.merge([h, s, i])

    def hsi_to_rgb(self, hsi_image):
        h, s, i = cv2.split(hsi_image)

        h = h * 2 * np.pi

        r = np.zeros_like(h)
        g = np.zeros_like(h)
        b = np.zeros_like(h)

        mask_1 = (0 <= h) & (h < 2 * np.pi / 3)
        mask_2 = (2 * np.pi / 3 <= h) & (h < 4 * np.pi / 3)
        mask_3 = (4 * np.pi / 3 <= h) & (h < 2 * np.pi)

        b[mask_1] = i[mask_1] * (1 - s[mask_1])
        r[mask_1] = i[mask_1] * (1 + s[mask_1] * np.cos(h[mask_1]) / np.cos(np.pi / 3 - h[mask_1]))
        g[mask_1] = 3 * i[mask_1] - (r[mask_1] + b[mask_1])

        h[mask_2] -= 2 * np.pi / 3
        r[mask_2] = i[mask_2] * (1 - s[mask_2])
        g[mask_2] = i[mask_2] * (1 + s[mask_2] * np.cos(h[mask_2]) / np.cos(np.pi / 3 - h[mask_2]))
        b[mask_2] = 3 * i[mask_2] - (r[mask_2] + g[mask_2])

        h[mask_3] -= 4 * np.pi / 3
        g[mask_3] = i[mask_3] * (1 - s[mask_3])
        b[mask_3] = i[mask_3] * (1 + s[mask_3] * np.cos(h[mask_3]) / np.cos(np.pi / 3 - h[mask_3]))
        r[mask_3] = 3 * i[mask_3] - (g[mask_3] + b[mask_3])

        return np.clip(cv2.merge([r, g, b]) * 255, 0, 255).astype(np.uint8)