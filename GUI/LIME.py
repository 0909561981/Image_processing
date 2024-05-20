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
from skimage.color import gray2rgb

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
    def HE_enhance(self):
        gray_image = rgb2gray(self.L)
        enhanced_gray_image = equalize_hist(gray_image)
        self.R = np.zeros(self.L.shape)
        for i in range(3):
            nonzero_mask = gray_image != 0
            self.R[:, :, i] = np.where(nonzero_mask, self.L[:, :, i] * enhanced_gray_image / np.where(nonzero_mask, gray_image, 1), self.L[:, :, i])
        self.R = rescale_intensity(self.R, (0, 1))
        self.R = img_as_ubyte(self.R)
        return self.R

    def HSI_enhance(self):
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
    
    def GC_enhance(self):
        gray_image = rgb2gray(self.L)
        gamma = 0
        for r in gray_image:
            for c in r:
                gamma += (c / (self.row * self.col))
        gamma *= 2
        enhanced_gray_image = gray_image ** gamma

        self.R = np.zeros(self.L.shape)
        for i in range(3):
            nonzero_mask = gray_image != 0
            self.R[:, :, i] = np.where(nonzero_mask, self.L[:, :, i] * enhanced_gray_image / np.where(nonzero_mask, gray_image, 1), self.L[:, :, i])
        self.R = rescale_intensity(self.R, (0, 1))
        return self.R
    
    def CVC_enhance(self):
        sq = 7
        alpha = 0.33
        beta = 0.33
        gamma = 0.33
        gray_image = rgb2gray(self.L)
        gray_image = gray_image * 255
        gray_image = gray_image.astype(int)
        new_gray_image = [[0]*self.col for _ in range(self.row)]
        xhis = self.Hx(image=gray_image, sq=sq)
        uhis = self.Hu()
        this = self.Ht(alpha=alpha, beta=beta, gamma=gamma, xhist=xhis, uhist=uhis)

        Px = [-1 for _ in range(256)]
        Pt = [-1 for _ in range(256)]
        newGrayLevel = [0 for _ in range(256)]
        for i in range( 256 ):
            newGrayLevel[i] = self.grayLevel(m=i, xhist=xhis, thist=this, Px=Px, Pt=Pt)

        for i in range( self.row ):
            for j in range( self.col ):
                original = gray_image[i][j]
                new_gray_image[i][j] = newGrayLevel[original]
                
        self.R = np.zeros(self.L.shape)
        for i in range(3):
            nonzero_mask = gray_image != 0
            self.R[:, :, i] = np.where(nonzero_mask, self.L[:, :, i] * new_gray_image / np.where(nonzero_mask, gray_image, 1), self.L[:, :, i])
        self.R = rescale_intensity(self.R, (0, 1))
        return self.R
    
    def hp(self, xm, xn):
        ret = (abs(xm-xn)+1) / 256
        return ret
    
    def Hx(self, image, sq):
        hist = [[0]*256 for _ in range(256)]
        for i in range( self.row ):
            for j in range( self.col ):
                for k in range( -int(sq/2), int(sq/2)+1 ):
                    for q in range( -int(sq/2), int(sq/2)+1 ):
                        if i+k >= 0 and i+k < self.row and j+q >= 0 and j+q < self.col:
                            hist[image[i][j]][image[i+k][j+q]] += 1
        total = 0
        for i in range( 256 ):
            for j in range( 256 ):
                hist[i][j] *= self.hp(i, j)
                total += hist[i][j]
        for i in range( 256 ):
            for j in range( 256 ):
                hist[i][j] /= total
        return hist
    
    def CDF(self, hist, m, saved):
        ret = 0
        if saved[m-1] != -1:
            ret = saved[m-1]
            for i in range( m ):
                ret += hist[m-1][i]
                ret += hist[i][m-1]
            ret -= hist[m-1][m-1]
        else:
            for i in range( m ):
                for j in range( m ):
                    ret += hist[i][j]
        return ret
    
    def Hu(self):
        uniform = 1 / (256*256)
        hist = [[uniform]*256 for _ in range(256)]
        return hist
    
    def theta(self, k, r0, thetas):
        if thetas[k-1] == -1:
            thetas[k-1] = self.theta(k=k-1, r0=r0, thetas=thetas)
        if thetas[k-2] == -1:
            thetas[k-2] = self.theta(k=k-2, r0=r0, thetas=thetas)
        if k < 257:
            return (r0 * thetas[k-1] - thetas[k-2])
        else:
            return ((1+r0) * thetas[k-1] - thetas[k-2])
    
    def phi(self, k, r0, phis):
        if phis[k+1] == -1:
            phis[k+1] = self.phi(k=k+1, r0=r0, phis=phis)
        if phis[k+2] == -1:
            phis[k+2] = self.phi(k=k+2, r0=r0, phis=phis)
        if k < 255:
            return (r0 * phis[k+1] - phis[k+2])
        else:
            return ((1+r0) * phis[k+1] - phis[k+2])
            
    def S_mat(self, alpha, beta, gamma):
        mat = [[0]*256 for _ in range(256)]
        r0 = (2*gamma + alpha + beta) / (-gamma)
        thetas = [-1 for _ in range(258)]
        thetas[0] = 0
        thetas[1] = 1
        for i in range( 256 ):
            if thetas[i+2] == -1:
                thetas[i+2] = self.theta(k=i+2, r0=r0, thetas=thetas)
        phis = [-1 for _ in range(258)]
        phis[256] = 1
        phis[257] = 0
        for i in range( 256 ):
            if phis[255-i] == -1:
                phis[255-i] = self.phi(k=255-i, r0=r0, phis=phis)
        for i in range( 256 ):
            for j in range( 256 ):
                if i < j:
                    mat[i][j] = ((-1)**(i+j) / (-gamma)) * (thetas[i+1]*phis[j+1] / thetas[257])
                elif i == j:
                    mat[i][j] = (1 / (-gamma)) * (thetas[i+1]*phis[i+1] / thetas[257])
                else:
                    mat[i][j] = ((-1)**(i+j) / (-gamma)) * (thetas[j+1]*phis[i+1] / thetas[257])
        return mat
    
    def Ht(self, alpha, beta, gamma, xhist, uhist):
        Smat = self.S_mat(alpha=alpha, beta=beta, gamma=gamma)
        hist = [[0] * 256 for _ in range(256)]
        for i in range( 256 ):
            for j in range( 256 ):
                hist[i][j] = alpha * xhist[i][j] + beta * uhist[i][j]
        thist = [[0]*256 for _ in range(256)]
        total = 0
        for i in range( 256 ):
            for j in range( 256 ):
                for k in range( 256 ):
                    thist[i][j] += (Smat[i][k] * hist[k][j])
                total += thist[i][j]
        for i in range( 256 ):
            for j in range( 256 ):
                thist[i][j] /= total
        return thist
    
    def grayLevel(self, m, xhist, thist, Px, Pt):
        if Px[m] == -1:
            Px[m] = self.CDF(hist=xhist, m=m, saved=Px)
        if Pt[0] == -1:
            Pt[0] = self.CDF(hist=thist, m=0, saved=Pt)
        min = abs(Px[m] - Pt[0])
        mini = 0
        for i in range( 1, 256 ):
            if Pt[i] == -1:
                Pt[i] = self.CDF(hist=thist, m=i, saved=Pt)
            temp = abs(Px[m] - Pt[i])
            if temp < min:
                min = temp
                mini = i
        return mini