
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D

############ Functions ############


# Draws a bounding box given center point and size
def draw_boxes(img, box_centers, size=(64, 64), color=(0, 0, 255), thick=2):
    imcopy = np.copy(img)
    for center in box_centers:
        cv2.rectangle(
            imcopy,
            (center[0] - size[0] // 2, center[1] - size[1] // 2),
            (center[0] + size[0] // 2, center[1] + size[1] // 2),
            color=color,
            thickness=thick)
    return imcopy


# Draw label boxes based on given labels
def draw_labeled_bboxes(img, labels, color=(0, 0, 255), thick=2):
    for bbox in range(1, labels[1] + 1):
        nonzero = (labels[0] == bbox).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        b = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        cv2.rectangle(img, b[0], b[1], color, thick)
        x = b[0][0]
        y = b[0][1]
        cv2.fillPoly(img, np.array([[(x, y), (x+100, y), (x+90, y+20), (x, y+20)]]), color)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "car {:02}".format(bbox), (x + 5, y + 15), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
    return img


# Plots image pixels in 3d color space
def plot3d(pixels, colors_rgb, axis_labels=list("RGB"), axis_limits=[(0, 255), (0, 255), (0, 255)]):
    # Create figure and 3D axes
    fig = plt.figure(figsize=(8, 8))
    ax = Axes3D(fig)

    # Set axis limits
    ax.set_xlim(*axis_limits[0])
    ax.set_ylim(*axis_limits[1])
    ax.set_zlim(*axis_limits[2])

    # Set axis labels and sizes
    ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
    ax.set_xlabel(axis_labels[0], fontsize=16, labelpad=16)
    ax.set_ylabel(axis_labels[1], fontsize=16, labelpad=16)
    ax.set_zlabel(axis_labels[2], fontsize=16, labelpad=16)

    # Plot pixel values with colors given in colors_rgb
    ax.scatter(
        pixels[:, :, 0].ravel(),
        pixels[:, :, 1].ravel(),
        pixels[:, :, 2].ravel(),
        c=colors_rgb.reshape((-1, 3)), edgecolors='none')
    return ax


############ Main logic ###########


if __name__ == '__main__':
    # Read sample image for visualization
    import feature_extraction as feat
    rgb = feat.read_image('data/cars/GTI_Right/image0003.png', 'RGB')

    '''
    rh, gh, bh, bincen, feature_vec = feat.HistogramFeatureExtractor().get_features(rgb, vis=True)

    # Plot a figure with all three bar charts
    if rh is not None:
        fig = plt.figure(figsize=(12, 3))
        plt.subplot(131)
        plt.bar(bincen, rh[0])
        plt.xlim(0, 256)
        plt.title('R Histogram')
        plt.subplot(132)
        plt.bar(bincen, gh[0])
        plt.xlim(0, 256)
        plt.title('G Histogram')
        plt.subplot(133)
        plt.bar(bincen, bh[0])
        plt.xlim(0, 256)
        plt.title('B Histogram')
        fig.tight_layout()
        plt.show()
    else:
        print('Your function is returning None for at least one variable...')

    # Select a small fraction of pixels to plot by subsampling it
    scale = max(rgb.shape[0], rgb.shape[1], 64) / 64  # at most 64 rows and columns
    img_small = cv2.resize(rgb, (np.int(rgb.shape[1] / scale), np.int(rgb.shape[0] / scale)), interpolation=cv2.INTER_NEAREST)

    # Convert subsampled image to desired color space(s)
    img_small_RGB = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR, matplotlib likes RGB
    img_small_HSV = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
    img_small_YUV = cv2.cvtColor(img_small, cv2.COLOR_BGR2YUV)
    img_small_rgb = img_small_RGB / 255.  # scaled to [0, 1], only for plotting

    # Plot and show
    plot3d(img_small_RGB, img_small_rgb)
    plt.show()
    plot3d(img_small_HSV, img_small_rgb, axis_labels=list("HSV"))
    plt.show()
    plot3d(img_small_YUV, img_small_rgb, axis_labels=list("YUV"))
    plt.show()

    feature_vec = feat.SpatialFeatureExtractor().get_features(rgb)
    plt.plot(feature_vec)
    plt.title('Spatially Binned Features')
    plt.show()
    '''

    feat_ex = feat.HogFeatureExtractor(orientation=11, pix_per_cell=16, cells_per_block=2, hog_channel=0)
    feature_vec = feat_ex.get_all_features(rgb[:, :, 0], [(0, 0)])
    print(feature_vec)
    feature_vec, vis_img = feat_ex.get_features(rgb, vis=True)
    print()
    print()
    print(feature_vec)
    plt.imshow(vis_img)
    plt.title('HOG Features')
    plt.show()
