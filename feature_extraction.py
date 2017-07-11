
import cv2
import numpy as np
import matplotlib.image as mpimg
from skimage.feature import hog


############ Functions ############


# Read an image file given desired color space
def read_image(file, color_space='RGB'):
    image = mpimg.imread(file)
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    return convert_image(image, color_space)


# Converts an image file into desired color space
def convert_image(image, color_space='RGB'):
    if color_space != 'RGB':
        if color_space == 'HSV':
            return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            return cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    return image


############# Classes #############


# Wrapper for spatial feature extraction from images
class SpatialFeatureExtractor:

    # Constructs extractor with given parameters
    def __init__(self, size=(32, 32)):
        self.size = size

    # Calculates spatial features and visualization
    def get_features(self, image, vis=False):
        features = cv2.resize(image, self.size).ravel()
        return features


# Wrapper for histogram feature extraction from images
class HistogramFeatureExtractor:

    # Constructs extractor with given parameters
    def __init__(self, nbins=32, bins_range=(0, 256)):
        self.nbins = nbins
        self.bins_range = bins_range

    # Calculates spatial features and visualization
    def get_features(self, image, vis=False):
        rhist = np.histogram(image[:, :, 0], bins=self.nbins, range=self.bins_range)
        ghist = np.histogram(image[:, :, 1], bins=self.nbins, range=self.bins_range)
        bhist = np.histogram(image[:, :, 2], bins=self.nbins, range=self.bins_range)
        hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
        if vis:
            # Generating bin centers
            bin_edges = rhist[1]
            bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges) - 1]) / 2
            return rhist, ghist, bhist, bin_centers, hist_features
        return hist_features


# Wrapper for hog feature extraction from images
class HogFeatureExtractor:

    # Constructs extractor with given parameters
    def __init__(
            self,
            orientation=9,
            pix_per_cell=8,
            cells_per_block=2,
            hog_channel=0):
        self.orientation = orientation
        self.pix_per_cell = pix_per_cell
        self.cells_per_block = cells_per_block
        self.hog_channel = hog_channel
        self.hog = cv2.HOGDescriptor(
            _winSize=(64, 64),
            _blockSize=(pix_per_cell * cells_per_block, pix_per_cell * cells_per_block),
            _blockStride=((pix_per_cell * cells_per_block) // 2, (pix_per_cell * cells_per_block) // 2),
            _cellSize=(pix_per_cell, pix_per_cell),
            _nbins=orientation,
            _derivAperture=1,       # Sobel kernel size
            _winSigma=-1,           # Gaussian kernel, -1 = disable
            _histogramNormType=0,   # block normalization method, 0 = L2Hys
            _L2HysThreshold=0.2,
            _gammaCorrection=True,
            _nlevels=64)


    # Calculates HOG features for all locations at once
    def get_all_features(self, image, locations):
        if self.hog_channel == 'ALL':
            hog_features = (
                self.hog.compute(
                    image[:, :, 0],
                    (self.pix_per_cell, self.pix_per_cell),
                    (0, 0),
                    locations=locations).reshape([len(locations), -1]),
                self.hog.compute(
                    image[:, :, 1],
                    (self.pix_per_cell, self.pix_per_cell),
                    (0, 0),
                    locations=locations).reshape([len(locations), -1]),
                self.hog.compute(
                    image[:, :, 2],
                    (self.pix_per_cell, self.pix_per_cell),
                    (0, 0),
                    locations=locations).reshape([len(locations), -1]))
            return np.hstack(hog_features)
        else:
            return self.hog.compute(
                    image[:, :, self.hog_channel],
                    (self.pix_per_cell, self.pix_per_cell),
                    (0, 0),
                    locations=locations).reshape([len(locations), -1])

    # Calculates HOG features for image using opencv
    def get_cv2_hog(self, image):
        features = self.hog.compute(image, (self.pix_per_cell, self.pix_per_cell), (0, 0), locations=[(0, 0)])
        return features.flatten()


    # Calculates HOG features and visualization
    def get_features(self, image, vis=False):
        if self.hog_channel == 'ALL':
            hog_features = []
            for channel in range(image.shape[2]):
                features = self.get_cv2_hog(image[:, :, channel])
                '''
                features = hog(
                    image[:, :, channel],
                    orientations=self.orientation,
                    pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                    cells_per_block=(self.cells_per_block, self.cells_per_block),
                    block_norm='L2',
                    transform_sqrt=True,
                    visualise=False,
                    feature_vector=True)
                '''
                hog_features.extend(features)
            return np.array(hog_features)
        elif vis:
            features, hog_image = hog(
                image[:, :, self.hog_channel],
                orientations=self.orientation,
                pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                cells_per_block=(self.cells_per_block, self.cells_per_block),
                block_norm='L2',
                transform_sqrt=True,
                visualise=True,
                feature_vector=True)
            return features, hog_image
        else:
            features = self.get_cv2_hog(image[:, :, self.hog_channel])
            '''
            features = hog(
                image[:, :, self.hog_channel],
                orientations=self.orientation,
                pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                cells_per_block=(self.cells_per_block, self.cells_per_block),
                block_norm='L2',
                transform_sqrt=True,
                visualise=False,
                feature_vector=True)
            '''
            return features
