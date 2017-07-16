
import numpy as np
from scipy.ndimage.measurements import label


############ Functions ############


def get_window_centers(windows, size=(64, 64), scale=1.0):
    window_centers = []
    for window in windows:
        window_center = (
            int(np.round((window[0] + 0.5 * size[0]) * scale)),
            int(np.round((window[1] + 0.5 * size[1]) * scale)))
        window_centers.append(window_center)
    return window_centers


# Calculates a two-dimensional list of car search windows based on a list of desired scales
def get_car_search_windows(image_size, levels):
    windows_list = []
    for i in range(len(levels)):
        level_size = (int(np.round(image_size[0] * levels[i])), int(np.round(image_size[1] * levels[i])))
        level_y1 = int(np.round(level_size[0] * (1.0 - 0.4 - 0.1 * levels[i])))
        level_y2 = int(np.round(level_size[0] * (1.2 - 0.7 * levels[i])))
        windows = get_search_windows(
            level_size,
            [None, None],
            [level_y1, level_y2],
            (64, 64),
            (0.75, 0.75))
        windows_list.append(windows)
    return windows_list


# Calculates a list of search windows based on given range and overlap
def get_search_windows(
        image_size,
        x_start_stop=[None, None],
        y_start_stop=[None, None],
        xy_window=(64, 64),
        xy_overlap=(0.75, 0.75)):

    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = image_size[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = image_size[0]

    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]

    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))

    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
    ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
    nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
    ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)

    window_list = []
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            startx = xs * nx_pix_per_step + x_start_stop[0]
            #endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            #endy = starty + xy_window[1]
            window_list.append((startx, starty))

    return window_list


############# Classes #############


class HeatMap:
    # Constructs a new heat map
    def __init__(self, image_size):
        self.history = []
        self.map = np.zeros((image_size[0], image_size[1]), np.float32)

    # Adds low-pass filter to heat
    def filter_heat(self, heatmap, window_size=10):
        # Append x values
        self.history.append(heatmap)

        # Enforce history size
        if len(self.history) > window_size:
            self.history.pop(0)

        # Calculate mean based on history
        mean_map = np.zeros(self.map.shape)
        mean_count = 0
        for map in self.history:
            if map is not None:
                mean_map += map
                mean_count += 1
        mean_map /= mean_count
        return mean_map

    # Adds heat
    def add_heat(self, bboxes, size, amount):
        for bbox in bboxes:
            p1 = (bbox[0] - size[0] // 2, bbox[1] - size[1] // 2)
            p2 = (bbox[0] + size[0] // 2, bbox[1] + size[1] // 2)
            self.map[p1[1]:p2[1], p1[0]:p2[0]] += 3*amount
            #for gradient in range(6, 10, 2):
            #    dw = int(size[0] / gradient)
            #    dh = int(size[1] / gradient)
            #    self.map[p1[1]+dw:p2[1]-dw, p1[0]+dh:p2[0]-dh] += amount
        self.map = self.filter_heat(self.map)

    # Apply threshold
    def apply_threshold(self, threshold):
        max_heat = np.max(self.map)
        if max_heat > 0.0:
            self.map /= max_heat
        self.map[self.map < threshold] = 0.0

    def get_labels(self):
        return label(self.map)

