import numpy as np
from skimage import img_as_float
from skimage.color import rgb2lab
from skimage.segmentation import slic
from scipy.spatial.distance import cdist


class SaliencyFilters:
    """
    A Python implementation of Saliency Filters.

    Reference:
    Perazzi, F., Krahenbuhl, P., Pritch, Y., & Hornung, A. (2012, June).
    Saliency filters: Contrast based filtering for salient region detection.
    In Computer Vision and Pattern Recognition (CVPR), 2012
    IEEE Conference on (pp. 733-740). IEEE.
    """

    def __init__(self, uniqueness_sigma=0.25, distribution_sigma=20.0, k=6.0,
                 alpha=0.033, beta=0.033, n_segments=300):
        self.__uniq_sigma = uniqueness_sigma
        self.__dist_sigma = distribution_sigma
        self.__k = k
        self.__alpha = alpha
        self.__beta = beta
        self.__n_segments = n_segments

    ###########################################################################
    # Public methods
    ###########################################################################

    def compute_saliency(self, image):
        """
        input:
        - image in RGB uint8
        output:
        - saliency map in uint8
        """

        # 1 - abstraction process
        superpixels, lab_color, rgb_color, position = self.__abstraction(image)

        # 2 - compute element uniqueness
        uniqueness = self.__uniqueness(superpixels, lab_color, position)

        # 3 - compute element distribution
        distribution = self.__distribution(superpixels, lab_color, position)

        # 4 - saliency assignment
        saliency = self.__saliency(uniqueness, distribution, rgb_color,
                                   position)

        # construct saliency map
        result = np.zeros(image.shape[:2])
        for superpixel in np.unique(superpixels):
            mask = superpixels == superpixel
            result[mask] = saliency[superpixel]

        return result

    ###########################################################################
    # Private methods
    ###########################################################################

    def __generate_superpixels(self, image):
        return slic(image, n_segments=self.__n_segments,
                    enforce_connectivity=True, compactness=30.0,
                    convert2lab=False)

    def __normalize(self, array):
        return (array - array.min()) / (array.max() - array.min() + 1e-13)

    def __gaussian_weight(self, array, sigma):
        weight = np.exp(-cdist(array, array) ** 2 / (2 * sigma ** 2))
        weight /= weight.sum(axis=1)[:, None]
        return weight

    def __gaussian_weight2(self, array1, array2, alpha, beta):
        weight = np.exp(-0.5 * (alpha * cdist(array1, array1) ** 2 +
                                beta * cdist(array2, array2) ** 2))
        weight /= weight.sum(axis=1)[:, None]
        return weight

    def __abstraction(self, image):
        nrgb = img_as_float(image)
        lab = rgb2lab(nrgb)
        nlab = (lab + np.array([0, 128, 128])) / np.array([100, 255, 255])

        # generate superpixels
        superpixels = self.__generate_superpixels(lab)
        n_segments = superpixels.max() + 1

        # construct position matrix
        max_y, max_x = np.array(superpixels.shape) - 1
        x = np.linspace(0, max_x, image.shape[1]) / max_x
        y = np.linspace(0, max_y, image.shape[0]) / max_y
        position = np.dstack((np.meshgrid(x, y)))

        # compute mean color and position
        mean_lab = np.zeros((n_segments, 3))
        mean_rgb = np.zeros((n_segments, 3))
        mean_position = np.zeros((n_segments, 2))
        for superpixel in np.unique(superpixels):
            mask = superpixels == superpixel
            mean_lab[superpixel, :] = nlab[mask, :].mean(axis=0)
            mean_rgb[superpixel, :] = nrgb[mask, :].mean(axis=0)
            mean_position[superpixel, :] = position[mask, :].mean(axis=0)

        return superpixels, mean_lab, mean_rgb, mean_position

    def __uniqueness(self, superpixels, color, position):
        weight = self.__gaussian_weight(position, self.__uniq_sigma)
        uniqueness = (cdist(color, color) ** 2 * weight).sum(axis=1)
        return self.__normalize(uniqueness)

    def __distribution(self, superpixels, color, position):
        weight = self.__gaussian_weight(color, self.__dist_sigma)
        mu = np.dot(weight, position)
        distribution = np.einsum('ij,ji->i', weight, cdist(position, mu) ** 2)
        return self.__normalize(distribution)

    def __saliency(self, uniqueness, distribution, color, position):
        saliency = uniqueness * np.exp(-self.__k * distribution)
        weight = self.__gaussian_weight2(color, position, self.__alpha,
                                         self.__beta)
        weighted_saliency = np.dot(weight, saliency)
        return self.__normalize(weighted_saliency)
