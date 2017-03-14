import cv2
import numpy as np
import os
from image_process import *

def get_all_train_image(path):
    all_file = os.listdir(path)
    all_train_image = []
    for image_name in all_file:
        image = cv2.imread(path + "/" + image_name, 0)
        all_train_image.append(cv2.resize(image, (25, 25)))
    return all_train_image
def get_hog() :
    winSize = (20,20)
    blockSize = (10,10)
    blockStride = (5,5)
    cellSize = (10,10)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradient = True

    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradient)

    return hog

def get_feature(image):
    image = cv2.resize(image, (25, 25))
    hog = get_hog()
    return hog.compute(image)

def get_classify_type(source_image, all_train_image):
    source_image_feature = get_feature(source_image);
    score = []
    for train_image in all_train_image:
        train_image_feature = get_feature(train_image)
        calculate_score = np.sum(np.square(train_image_feature-source_image_feature))
        score.append(calculate_score)
    score = np.array(score)
    result_list = [0, 1, 2, 3, 4, 6, 8, 9]
    min_index = result_list[np.argmin(score)]
    return min_index

def split2d(img, cell_size, flatten=True):
    h, w = img.shape[:2]
    sx, sy = cell_size
    cells = [np.hsplit(row, w//sx) for row in np.vsplit(img, h//sy)]
    cells = np.array(cells)
    if flatten:
        cells = cells.reshape(-1, sy, sx)
    return cells

if __name__ == '__main__':
    train_path = 'train'
    test_path = 'test'

    all_train_image = get_all_train_image(train_path)

    test_image = test_image_source = cv2.imread(test_path + '/smaller.bmp', 0)
    test_image = test_image_source = cv2.resize(test_image, (274, 73))
    # test_image = cv2.GaussianBlur(test_image,(3,3),20)
    # test_image = cv2.blur(test_image, (4, 4))
    cv2.imshow("image", test_image)
    cv2.waitKey()
    # test_image = test_image_source = cv2.resize(test_image, (test_image.shape[1], test_image.shape[0]))
    ## canny and dilate to get
    test_image = cv2.Canny(test_image, 43, 113)
    cv2.imshow("image", test_image)
    cv2.waitKey()
    white_test_image = np.zeros((test_image.shape))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    test_image = cv2.dilate(test_image, kernel)

    derp, contours, hierarchy = cv2.findContours(test_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(white_test_image, contours, -1, (255, 255, 255), 3)
    print len(contours)
    all_rect_filter_width = []
    for i in range(len(contours)):
        rect = cv2.boundingRect(contours[i])
        print rect

        # box = np.int0(cv2.boxPoints(rect))
        area = cv2.contourArea(contours[i])
        if area > 200:
            # filter width
            if (rect[2] > test_image_source.shape[1]/5):
                number = rect[2]/(test_image_source.shape[1]/8) + 1
                number = 2
                for j in xrange(number):
                    rect_new = (rect[0] + rect[2]/ number* j, rect[1], rect[2]/number, rect[3])

                    all_rect_filter_width.append(rect_new)
            else:
                all_rect_filter_width.append(rect)

    all_rect_filter_height = []
    for rect in all_rect_filter_width:
        if rect[3] > test_image_source.shape[0]/2:
            for k in xrange(2):
                rect_new = (rect[0], rect[1] + rect[3] / 2 * k, rect[2], 36)
                all_rect_filter_height.append(rect_new)
        else:
            rect_new = (rect[0], rect[1], rect[2], 36)
            all_rect_filter_height.append(rect_new)
    all_cell = []
    for rect in all_rect_filter_height:
        cell = test_image_source[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
        cell = cv2.resize(cell, (25, 25))
        result = get_classify_type(cell, all_train_image)
        cv2.rectangle(test_image_source, (rect[0], rect[1]), (rect[2] + rect[0], rect[3] + rect[1]),
                      (255, 255, 255), 1)
        cv2.putText(test_image_source, str(result), (rect[0], rect[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1)
        all_cell.append(cell)
    cv2.imshow("image", test_image_source)
    cv2.waitKey()
    cv2.destrpyAllWindows()
    ## split to get
    # test_image = cv2.resize(test_image, (272, 72))
    # cells = split2d(test_image, (34, 36))
    # for i in range(cells.shape[0]):
    #     cv2.imshow("image", cells[i])
    #     cv2.waitKey()
    # print cells

