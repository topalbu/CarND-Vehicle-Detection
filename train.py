import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog

from lesson_functions import *
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split

import pickle
import os
import matplotlib.pyplot as plt
from PIL import Image



#get directory of the car images
car_images_path = os.getcwd()+'/vehicles'

#load the car images data
cars = [os.path.join(dirpath, f)
    for dirpath, dirnames, files in os.walk(car_images_path)
    for f in files if f.endswith('.png')]

#get directory of the non-car images
non_car_images_path = os.getcwd()+'/non-vehicles'

#load the non-car images data
notcars = [os.path.join(dirpath, f)
    for dirpath, dirnames, files in os.walk(non_car_images_path)
    for f in files if f.endswith('.png')]

print('Number of vehicle data : ', len(cars))
print('Number of non-vehicle data : ', len(notcars))

# Visualize data example
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(Image.open(cars[0]))
ax1.set_title('Example of Car Image', fontsize=30)
ax2.imshow(Image.open(notcars[0]))
ax2.set_title('Example of Not Car Image', fontsize=30)
plt.show()





# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # 1) Define an empty list to receive features
    img_features = []
    # 2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)
    # 3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # 4) Append features to list
        img_features.append(spatial_features)
    # 5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        # 6) Append features to list
        img_features.append(hist_features)
    # 7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # 8) Append features to list
        img_features.append(hog_features)

    # 9) Return concatenated array of features
    return np.concatenate(img_features)


# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32,
                   hist_range=(0, 256), orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, spatial_feat=True,
                   hist_feat=True, hog_feat=True):
    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # 4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)
        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # 6) Predict using your classifier
        prediction = clf.predict(test_features)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # 8) Return windows for positive detections
    return on_windows


# Uncomment the following line if you extracted training
# data from .png images (scaled 0 to 1 by mpimg) and the
# image you are searching is a .jpg (scaled 0 to 255)
# image = image.astype(np.float32)/255
def create_search_windows(search_windows,image):
    '''
    Method to create search windows list
    :param search_windows: list to append the search windows
    :param image: image to be search on
    :return: None
    '''
    search_windows += slide_window(image.shape, x_start_stop=[None,None],
                                        y_start_stop=[400, 500],
                                        xy_window=(80, 80), xy_overlap=(0.75, 0.75))
    search_windows += slide_window(image.shape, x_start_stop=[None, None],
                                        y_start_stop=[400, 500],
                                        xy_window=(100, 100), xy_overlap=(0.75, 0.75))
    search_windows += slide_window(image.shape, x_start_stop=[None, None],
                                        y_start_stop=[420, 660],
                                        xy_window=(120, 120), xy_overlap=(0.5, 0.5))
    search_windows += slide_window(image.shape, x_start_stop=[None, None],
                                        y_start_stop=[500, 690],
                                        xy_window=(160, 160), xy_overlap=(0.5, 0.5))

def train_test_model(parameters,do_training=False,model_path = 'models/linear_svc.p', image = None):
    '''
    Method to train and test the model
    :param parameters: a dict that contains traing parameters (hog features, bin size, spatial szie etc.)
    :param do_training: a flag if we want to do training
    :param model_path: path to save model in
    :param image:  image file to test the model
    :return: image contains the boxes on the car position
    '''

    #cehck the model files exist if not do trainig or if the do_training is True restart train'ng the ex'st'ng model
    if not os.path.isfile(model_path) or do_training:
        print('Train the model.')
        t = time.time()
        # Extract the car features and create the feature vector
        car_features = extract_features(cars, color_space=color_space,
                                        spatial_size=spatial_size, hist_bins=hist_bins,
                                        orient=orient, pix_per_cell=pix_per_cell,
                                        cell_per_block=cell_per_block,
                                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                                        hist_feat=hist_feat, hog_feat=hog_feat)

        # Extract the notcar features and create the feature vector
        notcar_features = extract_features(notcars, color_space=color_space,
                                           spatial_size=spatial_size, hist_bins=hist_bins,
                                           orient=orient, pix_per_cell=pix_per_cell,
                                           cell_per_block=cell_per_block,
                                           hog_channel=hog_channel, spatial_feat=spatial_feat,
                                           hist_feat=hist_feat, hog_feat=hog_feat)

        #combine and convert the type of the feture vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.2, random_state=rand_state)

        print('Parameters:',parameters)
        print('Feature vector length:', len(X_train[0]))
        # Use a linear SVC
        svc = LinearSVC()
        # Check the training time for the SVC

        svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
        # Check the prediction time for a single sample
        with open(model_path, 'wb') as model_file:
            pickle.dump({'svc': svc,'X_scaler': X_scaler,'parameters': parameters},model_file,pickle.HIGHEST_PROTOCOL)
        print('Model has saved to ' ,model_path)
    else:
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
            svc = model['svc']
            X_scaler = model['X_scaler']
            parameters = model['parameters']
    if image is not None:
        draw_image = np.copy(image)
        windows = []
        create_search_windows(windows, image)
        print('windows size : ', len(windows))

        image = image.astype(np.float32) / 255
        t = time.time()
        hot_windows = search_windows(image, windows, svc, X_scaler, color_space=parameters['color_space'],
                                        spatial_size=parameters['spatial_size'], hist_bins=parameters['hist_bins'],
                                        orient=parameters['orient'], pix_per_cell=parameters['pix_per_cell'],
                                        cell_per_block=parameters['cell_per_block'],
                                        hog_channel=parameters['hog_channel'], spatial_feat=parameters['spatial_feat'],
                                        hist_feat=parameters['hist_feat'], hog_feat=parameters['hog_feat'])

        window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to search_windows ...')
        return window_img



### TODO: Tweak these parameters and see how the results change.


color_space = 'RGB'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = 0  # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32)  # Spatial binning dimensions
hist_bins = 32  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off
y_start_stop = [400, 690]  # Min and max in y to search in slide_window()

color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 8  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16)  # Spatial binning dimensions
hist_bins = 16  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off
y_start_stop = [400, 690]  # Min and max in y to search in slide_window()



parameters = { 'color_space' : color_space,
               'orient': orient ,
               'pix_per_cell': pix_per_cell,
               'cell_per_block':cell_per_block,
               'hog_channel':hog_channel,
               'hist_bins':hist_bins,
               'spatial_feat':spatial_feat,
               'hist_feat':hist_feat,
               'hog_feat':hog_feat,
               'spatial_size':spatial_size }

image = mpimg.imread('test_images/test6.jpg')
window_image = train_test_model(parameters,do_training=True, model_path='models/linear_svc.p',image=image)

plt.imshow(window_image)
plt.show()