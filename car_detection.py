
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
import lane_detection


############ Functions ############


# Search a given list of windows using classifier
def search_windows(image, windows, classifier, scaler, feature_extractors):
    on_windows = []
    features = feature_extractors[0].get_all_features(image, windows)
    test_features = scaler.transform(features)
    prediction = classifier.predict(test_features)
    for i in range(len(prediction)):
        if prediction[i] == 1:
            on_windows.append(windows[i])
    return on_windows


# Extracts features from a single image
def extract_features(image, feature_extractors):
    img_features = []
    for extractor in feature_extractors:
        img_features.append(extractor.get_features(image))
    return np.concatenate(img_features)


# Extracts features from a list of image file
def extract_all_features(image_files, color_space, feature_extractors):
    features = []
    for image_file in image_files:
        image = feat.read_image(image_file, color_space)
        file_features = extract_features(image, feature_extractors)
        features.append(file_features)
    return features


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
    def __init__(self, camera_calibration, image_size, color_space, classifier, scaler, feature_extractors, img=None):
        self.subprocessor = lane_detection.ImageProcessor(camera_calibration, image_size, 0.375)
        self.image_size = image_size
        self.color_space = color_space
        self.classifier = classifier
        self.scaler = scaler
        self.feature_extractors = feature_extractors
        self.heat_map = wins.HeatMap(image_size)
        self.levels = [0.7, 0.6, 0.5, 0.4, 0.3]
        self.level_colors = [
            (255, 255, 255),
            (255, 255, 0),
            (255, 127, 0),
            (255, 0, 0),
            (255, 0, 255),
            (0, 0, 255),
            (0, 255, 255),
            (0, 255, 0)]
        self.windows = wins.get_car_search_windows(image_size, self.levels)
        if img is not None:
            for i in range(len(self.windows)):
                scale = 1.0/self.levels[i]
                w = wins.get_window_centers(self.windows[i], size=(64, 64), scale=scale)
                img = draw.draw_boxes(img, w, size=(int(64*scale), int(64*scale)), color=self.level_colors[i], thick=2)
            plt.imshow(img)
            plt.show()
        self.frame = 0


    # Implements the car marking pipeline
    def pipeline(self, img):
        self.frame += 1

        # Require defined image size
        assert img.shape == self.image_size

        # Undistort & detect lanes
        img = self.subprocessor.pipeline(img)

        # Match cars
        match_img = feat.convert_image(img, self.color_space)
        result_windows = []
        for i in range(len(self.levels)):
            scaled = cv2.resize(match_img, (0, 0), fx=self.levels[i], fy=self.levels[i])
            on_windows = search_windows(scaled, self.windows[i], self.classifier, self.scaler, self.feature_extractors)
            result_windows.append(wins.get_window_centers(on_windows, scale=1.0/self.levels[i]))

        # Add heat
        for i in range(len(self.levels)):
            scale = 1.0 / self.levels[i]
            self.heat_map.add_heat(result_windows[i], size=(int(64*scale), int(64*scale)), amount=0.05)
            #img = draw.draw_boxes(img, result_windows[i], size=(int(64 * scale), int(64 * scale)), color=self.level_colors[i], thick=2)
        self.heat_map.apply_threshold(0.2)

        # Get labels and draw boxes
        labels = self.heat_map.get_labels()
        img = draw.draw_labeled_bboxes(img, labels, color=(0, 100, 255), thick=4)

        ''' Generates pictures write writeup
        f, ax = plt.subplots(1, 2, figsize=(10, 3.5), frameon=False)
        f.subplots_adjust(hspace=0.15, wspace=0.00, left=0, bottom=0, right=1, top=0.97)
        ax[0].set_title('Frame {:02}'.format(self.frame))
        ax[0].axis('off')
        ax[0].imshow(img)
        ax[1].set_title('Labels {:02}'.format(self.frame))
        ax[1].axis('off')
        #ax[1].imshow(self.heat_map.map, cmap="magma")
        #ax[1].imshow(labels[0], cmap="gray")
        #plt.show()
        f.canvas.draw()
        data = np.fromstring(f.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(f.canvas.get_width_height()[::-1] + (3,))
        return data
        '''

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

    clf, X_scaler = load_classifier('data/trained_classifier.p')
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
        print(X.shape)
        X = X_scaler.transform(X)
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(no_car_features))))

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=rand_state)

        # Train
        print("Training..")
        parameters = {'kernel': ['rbf']}#, 'C': [0.7, 0.75, 0.8], 'max_iter': [1000], 'gamma': [0.975, 0.1, 0.125]}
        svr = svm.SVC()
        clf = GridSearchCV(svr, parameters, n_jobs=4)
        t = time.time()
        clf.fit(X_train, y_train)
        print('Training time: ', round(time.time() - t, 2))
        print('Best params  : ', clf.best_params_, 'best score: ', clf.best_score_)
        print('Test accuracy: ', clf.score(X_test, y_test))
        print('Test predicts: ', clf.predict(X_test[0:10]))
        print('For labels   : ', y_test[0:10])
        save_classifier('data/trained_classifier.p', clf, X_scaler)

    '''
    # Pipeline for pictures
    for img_file in glob.glob('test_images/*.jpg'):
        rgb = feat.read_image(img_file, color_space='RGB')
        processor = ImageProcessor(cam_calibration, rgb.shape, cspace, clf, X_scaler, used_extractors)
        t = time.time()
        rgb = processor.pipeline(rgb)
        mpimg.imsave('output_images/{}'.format(os.path.basename(img_file)), rgb, format='jpg')
        print('Prediction time: ', round(time.time() - t, 2))
        #plt.imshow(rgb, cmap='magma')
        #plt.show()
    '''
    # Pipeline for video
    clip = VideoFileClip('test_videos/project_video.mp4')
    processor = ImageProcessor(cam_calibration, (clip.h, clip.w, 3), cspace, clf, X_scaler, used_extractors)
    new_clip = clip.fl_image(lambda frame: processor.pipeline(frame))
    new_clip.write_videofile('output_videos/project_video.mp4', audio=False)

