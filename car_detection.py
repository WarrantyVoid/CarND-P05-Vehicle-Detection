
import cv2
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
import sklearn.svm as svm
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import calibration
import feature_extraction as feat
import window_search as wins
import visualization as draw
import os


############ Functions ############


# Extracts features from a list of image file
def extract_all_features(image_files, color_space, feature_extractors):
    features = []
    for image_file in image_files:
        image = feat.read_image(image_file, color_space)
        file_features = extract_features(image, feature_extractors)
        features.append(file_features)
    return features


# Search a given list of windows using classifier
def search_windows(image, windows, classifier, scaler, feature_extractors):
    on_windows = []
    for window in windows:
        test_img = cv2.resize(image[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        features = extract_features(test_img, feature_extractors)
        test_features = scaler.transform(features)
        prediction = classifier.predict(test_features)
        if prediction == 1:
            on_windows.append(window)
    return on_windows


# Extacts features from a single image
def extract_features(image, feature_extractors):
    img_features = []
    for extractor in feature_extractors:
        img_features.append(extractor.get_features(image))
    return np.concatenate(img_features)


# Loads classifier
def load_classifier(file_name):
    if os.path.isfile(file_name):
        return joblib.load(file_name)
    return None, None


# Stores classifier
def save_classifier(file_name, classifier, scaler):
    joblib.dump((classifier, scaler), file_name, compress=9)

############# Classes #############


# Represents a lane line processor for images
class ImageProcessor:
    # Constructs the processor with given image/frame size
    def __init__(self, camera_calibration, image_size, color_space, classifier, scaler, feature_extractors):
        self.camera_calibration = camera_calibration
        self.image_size = image_size
        self.color_space = color_space
        self.classifier = classifier
        self.scaler = scaler
        self.feature_extractors = feature_extractors
        self.windows = []
        windows0 = wins.get_search_windows(
            image_size,
            [int(image_size[1] * 0.3), int(image_size[1] * 0.7)],
            [int(image_size[0] * 0.55), int(image_size[0] * 0.65)],
            (32, 32),
            (0.75, 0.75))
        windows1 = wins.get_search_windows(
            image_size,
            [None, None],
            [int(image_size[0] * 0.55), int(image_size[0] * 0.75)],
            (100, 100),
            (0.75, 0.75))
        windows2 = wins.get_search_windows(
            image_size,
            [None, None],
            [int(image_size[0] * 0.6), int(image_size[0] * 0.85)],
            (140, 140), (0.75, 0.75))
        windows3 = wins.get_search_windows(
            image_size,
            [None, None],
            [int(image_size[0] * 0.65), int(image_size[0] * 1.0)],
            (180, 180),
            (0.75, 0.75))
        self.windows.extend(windows0)
        self.windows.extend(windows1)
        self.windows.extend(windows2)
        self.windows.extend(windows3)
        #img = draw.draw_boxes(img, windows0, color=(255, 0, 255))
        #img = draw.draw_boxes(img, windows1, color=(0, 0, 255))
        #img = draw.draw_boxes(img, windows2, color=(255, 0, 0))
        #img = draw.draw_boxes(img, windows3, color=(0, 255, 0))

    # Implements the car marking pipeline
    def pipeline(self, img):
        # Require defined image size
        assert img.shape == self.image_size

        # Undistort
        img = cam_calibration.undistort(img)

        # Match cars
        match_img = feat.convert_image(img, self.color_space)
        result_windows = search_windows(match_img, self.windows, self.classifier, self.scaler, self.feature_extractors)

        # Draw
        img = draw.draw_boxes(img, result_windows)
        return img


############ Main logic ###########


if __name__ == '__main__':
    cam_calibration = calibration.CameraCalibration()

    # Setup and parametrization of available feature extractors
    spatial_extract = feat.SpatialFeatureExtractor(size=(32, 32))
    histogram_extract = feat.HistogramFeatureExtractor(nbins=32, bins_range=(0, 256))
    hog_extract = feat.HogFeatureExtractor(orientation=11, pix_per_cell=16, cells_per_block=2, hog_channel='ALL')
    used_extractors = [hog_extract]
    cspace = 'YUV'

    clf, X_scaler = load_classifier('classifier.p')
    if clf is not None and X_scaler is not None:
        print("Classifier loaded.")
    else:
        # Read in car and non-car images
        print("Reading training images..")
        cars = []
        no_cars = []
        for car in glob.glob('data/cars/**/*.png', recursive=True):
            cars.append(car)
        for car in glob.glob('data/extras/**/*.png', recursive=True):
            no_cars.append(car)
        print("Cars: ", len(cars), ", No Cars: ", len(no_cars))
        cars = shuffle(cars)
        no_cars = shuffle(no_cars)
        min_len = min(len(cars), len(no_cars))
        cars = cars[:min_len]
        no_cars = no_cars[:min_len]

        # Define features & labels
        print("Extracting training samples..")
        car_features = extract_all_features(cars, cspace, used_extractors)
        no_car_features = extract_all_features(no_cars, cspace, used_extractors)
        X = np.vstack((car_features, no_car_features)).astype(np.float64)
        X_scaler = StandardScaler().fit(X)
        X = X_scaler.transform(X)
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(no_car_features))))

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=rand_state)

        # Train
        print("Training..")
        parameters = {'kernel': ['rbf'], 'C': [1], 'max_iter': [1000]}#, 'gamma': [0.06, 0.07, 0.08, 0.09, 0.1]}
        svr = svm.SVC()
        clf = GridSearchCV(svr, parameters, n_jobs=4)
        t = time.time()
        clf.fit(X_train, y_train)
        print('Training time: ', round(time.time() - t, 2))
        print('Best params  : ', clf.best_params_, 'best score: ', clf.best_score_)
        print('Test accuracy: ', clf.score(X_test, y_test))
        print('Test predicts: ', clf.predict(X_test[0:10]))
        print('For labels   : ', y_test[0:10])
        save_classifier('classifier.p', clf, X_scaler)

    '''
    # Pipeline for single picture
    rgb = feat.read_image('test_images/test6.jpg', color_space='RGB')
    processor = ImageProcessor(cam_calibration, rgb.shape, cspace, clf, X_scaler, used_extractors)
    t = time.time()
    plt.imshow(processor.pipeline(rgb))
    print('Prediction time: ', round(time.time() - t, 2))
    plt.show()

    '''
    # Pipeline for video
    clip = VideoFileClip('test_videos/project_video.mp4')
    processor = ImageProcessor(cam_calibration, (clip.h, clip.w, 3), cspace, clf, X_scaler, used_extractors)
    new_clip = clip.fl_image(lambda frame: processor.pipeline(frame))
    new_clip_output = 'output_videos/project_video.mp4'
    new_clip.write_videofile(new_clip_output, audio=False)
