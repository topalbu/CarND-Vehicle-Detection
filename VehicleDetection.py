import math
import matplotlib.pyplot as plt
import time
from lesson_functions import *
import pickle
from scipy.ndimage.measurements import label


class Box:
    # Box class to represent and keep track of the cars on a video stream
    def __init__(self, box):
        # x1 coordinate of the box
        self.startX = box[0][0]
        # y1 coordinate of the box
        self.startY = box[0][1]
        # x2 coordinate of the box
        self.endX = box[1][0]
        # y2 coordinate of the box
        self.endY = box[1][1]
        # center (x,y) coordinte
        self.center = self.calculate_center()
        # width (x2-x1) length
        self.width = self.calculate_width()
        # height (y2-y1) length
        self.height = self.calculate_height()
        # heat of the box (represent
        self.heat = 0
        # speed vector in x-y pixel coordinates
        self.speed = [0, 0]
        # age to track and get rid of old boxes (vehicles detected and dragged along the video frames)
        self.age = 0

    def __getitem__(self, key):
        """
            Return the value of start or end points of the box
            :param key to select start or end point
        """
        if key == 0:
            return (self.__getattribute__('startX'), self.__getattribute__('startY'))
        elif key == 1:
            return (self.__getattribute__('endX'), self.__getattribute__('endY'))

    def calculate_center(self):
        """
            Calculates the center coordinates of the box
        """
        return [(self.startX + self.endX) / 2., (self.startY + self.endY) / 2.]

    def calculate_distance(self, other):
        """
            Calculates center to center distance to given box
            :param other box to calculate distance
        """
        return math.sqrt((self.center[0] - other.center[0]) ** 2 + (self.center[1] - other.center[1]) ** 2)

    def is_equal(self, other):
        """
            Checks if the given box has similar attributes
            :param other box to compare
        """
        return self.close(other) & (abs(self.width - other.width) < 100) & (abs(self.height - other.height) < 100)

    def close(self, other):
        """
            Checks if the given box is closer in a certain threhold
            :param other box to compare
        """
        return (self.calculate_distance(other) < 200)

    def calculate_width(self):
        """
            Calculates width of the box
        """
        return self.endX - self.startX

    def calculate_height(self):
        """
            Calculates height of the box
        """
        return self.endY - self.startY

    def print_cords(self):
        """
            Prints the x and y coorinates of the box
        """
        print('startX  :', self.startX, ' ,startY  :', self.startY, ' ,endX  :', self.endX, ' ,endY  :', self.endY)

    # def merge(self, other):
    #     # new_center = [0,0]
    #     # new_center[0] = (self.center[0]*self.heat + other.center[0]*other.heat)/2.
    #     # new_center[1] = (self.center[1]*self.heat + other.center[1]*other.heat)/2.
    #     sum_heat = (self.heat + other.heat)
    #     self.startX = int((self.startX * self.heat + other.startX * other.heat) / (sum_heat * 1.01))
    #     self.startY = int((self.startY * self.heat + other.startY * other.heat) / (sum_heat * 1.01))
    #     self.endX = int((self.endX * self.heat + other.endX * other.heat) / (sum_heat * 0.99))
    #     self.endY = int((self.endY * self.heat + other.endY * other.heat) / (sum_heat * 0.99))
    #     self.center = self.calculate_center()
    #     self.width = self.endX - self.startX
    #     self.height = self.endY - self.startY
    #     self.heat = sum_heat / 2.
    #     del other


    def calculate_speed(self, old):
        """
            Calculates the speed in xy pixel coordinate wtih respect to old position
            :param old box from the previous frame
        """
        self.speed[0] = self.center[0] - old.center[0]
        self.speed[1] = self.center[1] - old.center[1]

    def _print(self):
        """
            Prints the attributes of the box
        """
        print('center  :', self.center, ' widht : ', self.width, ' height : ', self.height, ' heat : ', self.heat,
              ' speed ', self.speed)


# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='YCrCb',
                   spatial_size=(16, 16), hist_bins=16,
                   orient=8,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel='ALL', spatial_feat=True,
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


    # Define a function to draw bounding boxes


class VehicleDetector:
    """
    Class to detect vehicles on video stream it
    :param old_positions list to keep track of the cars located on previous frames
    :param car_list list to keep located cars on current frame
    :param scv classifier to classify windows whether it is car or not
    :param X_scaler standart scaler for normalization
    :param parameters feature parameters
    :param debug flag to enable/disable  debugging
    :param ımage_shape shape of the image to be work on
    """
    old_positions = []  # positions of detectefd vehicles on the last frame
    car_list = []
    svc = None
    X_scaler = None
    parameters = None
    debug = True
    image_shape = (720, 1280)

    def __init__(self, image):
        self.heat = np.zeros_like(image[:, :, 0]).astype(np.float)
        self.image = image
        self.search_windows = []
        self.create_search_windows()
        self.load_cls_params()
        self.image_shape = image.shape

    def load_cls_params(self):
        """
        Method to load classifier, scaler and feature parameters from file
        """
        with open('models/linear_svc.p', 'rb') as model_file:
            model = pickle.load(model_file)
            self.svc = model['svc']
            self.X_scaler = model['X_scaler']
            self.parameters = model['parameters']

        print(self.parameters)

    def age_heat_map(self):
        """
        Method to age old vehıcles by reducing heat
        """
        nonzero = self.heat.nonzero()
        self.heat[nonzero] -= 1

    def create_search_windows(self):
        self.search_windows += slide_window(self.image.shape, x_start_stop=[0, image.shape[1]],
                                            y_start_stop=[400, 500],
                                            xy_window=(80, 80), xy_overlap=(0.75, 0.75))
        self.search_windows += slide_window(self.image.shape, x_start_stop=[None, None],
                                            y_start_stop=[400, 500],
                                            xy_window=(100, 100), xy_overlap=(0.5, 0.5))
        self.search_windows += slide_window(self.image.shape, x_start_stop=[None, None],
                                            y_start_stop=[420, 660],
                                            xy_window=(120, 120), xy_overlap=(0.5, 0.5))
        self.search_windows += slide_window(self.image.shape, x_start_stop=[None, None],
                                            y_start_stop=[500, 690],
                                            xy_window=(160, 160), xy_overlap=(0.5, 0.5))

image = mpimg.imread('test_images/test1.jpg')

vehicleDetertor = VehicleDetector(image)
import glob


def add_heat(heatmap, bbox_list):
    """
    Iterates through the box list and increment the value of the pixel for each box if the corresponding pixel is on the box
    """
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap


def apply_threshold(heatmap, threshold):
    """
    :param heatmap: previous heatmap to be updated
    :param threshold:
    :return: updated heatmap
    """
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes 2

    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        if bbox.heat > 14:
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
            font = cv2.FONT_HERSHEY_SIMPLEX
            speed_text = "Heat: {0:.2f} ".format(bbox.heat)
            cv2.putText(imcopy, speed_text, (int(bbox.center[0]), int(bbox.center[1])), font, 1, (255, 255, 255), 2)
            # speed_text = "Y Speed: {0:.2f} ".format(bbox.speed[1])
            # cv2.putText(imcopy, speed_text, (int(bbox.center[0]),int(bbox.center[1]+20)), font, 1, (255, 255, 255), 2)
    # Return the image copy with boxes drawn
    return imcopy


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img


def locate_cars(img):
    # Iterate through all detected cars
    # Find final boxes from heatmap using label function
    # add_heat(img,vehicleDetertor.old_positions)
    # apply_threshold(img,len(vehicleDetertor.old_positions)-1)


    labels = label(img)
    car_list = []
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = Box(((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy))))
        bbox.heat = img[bbox.center[1]][bbox.center[0]]
        # if bbox.heat > 5:
        car_list.append(bbox)

    # Return car list

    for old_car in vehicleDetertor.old_positions:
        # print('old cars : ')
        # old_car._print()
        # print()
        found = False
        for car in car_list:
            # print('new cars : ')
            # car._print()
            if car.is_equal(old_car):
                # print('is  equal : ')
                car.calculate_speed(old_car)
                car.heat = max(car.heat, old_car.heat) + 1
                found = True
                break

        if not found and old_car.heat > 3 and old_car.age < 4:
            # print(' old car not found in the new frame but shoud be there :   ' ,old_car.heat)
            old_car.age += 1
            old_car.heat -= 1
            car_list.append(old_car)

            # print()
    return car_list

from multiprocessing import Pool
pool = Pool(4)
def process_image(image):
    copy_image = np.copy(image)
    copy_image = copy_image.astype(np.float32) / 255.

    A = (copy_image, vehicleDetertor.search_windows[0:50], vehicleDetertor.svc,
                                 vehicleDetertor.X_scaler)
    B = (copy_image, vehicleDetertor.search_windows[50:100], vehicleDetertor.svc,
                                 vehicleDetertor.X_scaler)

    C = (copy_image, vehicleDetertor.search_windows[100:150], vehicleDetertor.svc,
         vehicleDetertor.X_scaler)

    D = (copy_image, vehicleDetertor.search_windows[150:len(vehicleDetertor.search_windows)], vehicleDetertor.svc,
         vehicleDetertor.X_scaler)

    result1 = pool.apply_async(search_windows, A)  # evaluate "solve1(A)" asynchronously
    result2 = pool.apply_async(search_windows, B)  # evaluate "solve2(B)" asynchronously
    result3 = pool.apply_async(search_windows, C)  # evaluate "solve1(A)" asynchronously
    result4 = pool.apply_async(search_windows, D)  # evaluate "solve2(B)" asynchronously
    answer1 = result1.get(timeout=10)
    answer2 = result2.get(timeout=10)
    answer3 = result3.get(timeout=10)
    answer4 = result4.get(timeout=10)

    '''
    hot_windows = search_windows(copy_image, vehicleDetertor.search_windows, vehicleDetertor.svc,
                                 vehicleDetertor.X_scaler, color_space=vehicleDetertor.parameters['color_space'],
                                 spatial_size=vehicleDetertor.parameters['spatial_size'],
                                 hist_bins=vehicleDetertor.parameters['hist_bins'],
                                 orient=vehicleDetertor.parameters['orient'],
                                 pix_per_cell=vehicleDetertor.parameters['pix_per_cell'],
                                 cell_per_block=vehicleDetertor.parameters['cell_per_block'],
                                 hog_channel=vehicleDetertor.parameters['hog_channel'],
                                 spatial_feat=vehicleDetertor.parameters['spatial_feat'],
                                 hist_feat=vehicleDetertor.parameters['hist_feat'],
                                 hog_feat=vehicleDetertor.parameters['hog_feat'])
    '''
    hot_windows = answer1
    hot_windows += answer2
    hot_windows += answer3
    hot_windows += answer4
    heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    add_heat(heat, hot_windows)
    apply_threshold(heat, 1)
    # Visualize the heatmap when displaying
    # heatmap = np.clip(heat, 0, 255)

    VehicleDetector.old_positions = vehicleDetertor.car_list
    vehicleDetertor.car_list = locate_cars(heat)

    # labels = label(heat)
    # draw_img = draw_labeled_bboxes(image,labels)
    draw_img = draw_boxes(image, vehicleDetertor.car_list)
    return draw_img


import os
from moviepy.editor import VideoFileClip

white_output = 'white.mp4'  # New video
os.remove(white_output)
clip1 = VideoFileClip('project_video.mp4')  # .subclip(21.00,25.00) # project video
# clip = VideoFileClip("myHolidays.mp4", audio=True).subclip(50,60)
white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

# images = glob.glob('test_images/*.jpg')
# for image_path in images:
#     image = mpimg.imread(image_path)
#     t = time.time()
#     draw_image = process_image(image)
#     t2 = time.time()
#     print(round(t2 - t, 2), 'procesing time ...')
#     print('Numer Of Cars = in old frame : ' , len(vehicleDetertor.old_positions))
#     print('Numer Of Cars = in new frame : ', len(vehicleDetertor.car_list))
#     vehicleDetertor.old_positions = vehicleDetertor.car_list
#     fig = plt.figure()
#     plt.subplot(121)
#     plt.imshow(image)
#     plt.title('Car Positions')
#     plt.subplot(122)
#     plt.imshow(draw_image, cmap='hot')
#     plt.title('Heat Map')
#     fig.tight_layout()
#     plt.show()
