#coding=utf-8
import cv2
import numpy as np
import math
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

def construct_image(image, ratio):
    image_left = crop_image_by_ratio(image, (0, 1, 0, ratio[0]))
    image_right = crop_image_by_ratio(image, (0, 1, ratio[1], 1))
    image_new = np.concatenate([image_left, image_right], axis=1)
    return image_new

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def get_position_with_circle(img, bound_radius=(50,100), kernel=9, iterations=2, canny_bound=70, center_bound=(0, 1, 0, 1)):
    # kernal must be larger than 7 here
    ret , binary = cv2.threshold(canny_image(img, kernel=kernel, bound=canny_bound), 250, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(binary , kernel , iterations=iterations)
    # cv2.imshow("Dialated", dilated)
    # cv2.waitKey(37)
    circles = cv2.HoughCircles(dilated , cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50 , param2=30, minRadius=bound_radius[0], maxRadius=bound_radius[1])
    all_circle = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            print i
            if i[1] < center_bound[1] and i[1] > center_bound[0]:
                all_circle.append(i)

    return all_circle

def rotate_about_center(src, angle, scale=1.):
    """
    rotate image base on center
    :param src:
    :param angle:
    :param scale:
    :return: rotate image
    """
    w = src.shape[1]
    h = src.shape[0]
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]
    return cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

def canny_image(image, kernel=11, bound=50):
    """

    :param image: gray image
    :param kernel:
    :return: canny image
    """
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(image, (kernel, kernel), 200)
    # img = gray
    canny = cv2.Canny(img, bound, bound*3)
    return canny

def get_position(img, bound_area=1000, iterations=2, kernel=9):
    """
    get objects in an image
    :param img: source image
    :param bound_area: the min area of objects
    :param iterations: dilate iterations
    :param kernel: canny kernel
    :return:a list that contains all (box, rect) pairs
    """
    # kernal must be larger than 7 here
    ret , binary = cv2.threshold(canny_image(img, kernel=kernel), 250, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(binary, kernel, iterations=iterations)
    # cv2.imshow("Dialated", dilated)
    # cv2.waitKey()
    derp, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # c = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    all_contours = []
    for i in range(len(contours)):
        rect = cv2.minAreaRect(contours[i])
        box = np.int0(cv2.boxPoints(rect))
        area = cv2.contourArea(contours[i])
        if area > bound_area:
            all_contours.append((box, rect))
        # cv2.drawContours(img , [box] , -1 , (100 , 100 , 100) , 6)
    # cv2.imshow("Img", img)
    # cv2.waitKey()

    return all_contours

def filter_matches(kp1, kp2, matches, ratio = 0.75):
    """
    filter matches points
    :param kp1: kp1,des1 = sift.detectAndCompute(img1_gray, None) can be sift or surf etc
    :param kp2: kp2,des2 = sift.detectAndCompute(img1_gray2, None) can be sift or surf etc
    :param matches: match points
    :param ratio: filter ratio
    :return:feature point of image 1 and image 2 and key point pairs
    """
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append( kp1[m.queryIdx] )
            mkp2.append( kp2[m.trainIdx] )
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, kp_pairs

def explore_match(win, img1, img2, kp_pairs, status = None, H = None):
    """
    show match points between two images
    :param win:
    :param img1: image 1
    :param img2: image 2
    :param kp_pairs: can get by sift or surf etc
    :param status:
    :param H: transform matrix
    :return:show image
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    if H is not None:
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        # img2 = cv2.warpPerspective(img2, H, (img2.shape[1], img2.shape[0]))
        # cv2.imshow('jj', img2)
        # cv2.waitKey()
        corners = np.int32( cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0) )
        cv2.polylines(vis, [corners], True, (255, 255, 255))

    if status is None:
        status = np.ones(len(kp_pairs), np.bool)
    p1 = np.int32([kpp[0].pt for kpp in kp_pairs])
    p2 = np.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0)

    green = (0, 255, 0)
    red = (0, 0, 255)
    white = (255, 255, 255)
    kp_color = (51, 103, 236)
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            col = green
            cv2.circle(vis, (x1, y1), 2, col, -1)
            cv2.circle(vis, (x2, y2), 2, col, -1)
        else:
            col = red
            r = 2
            thickness = 3
            cv2.line(vis, (x1-r, y1-r), (x1+r, y1+r), col, thickness)
            cv2.line(vis, (x1-r, y1+r), (x1+r, y1-r), col, thickness)
            cv2.line(vis, (x2-r, y2-r), (x2+r, y2+r), col, thickness)
            cv2.line(vis, (x2-r, y2+r), (x2+r, y2-r), col, thickness)
    vis0 = vis.copy()
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            cv2.line(vis, (x1, y1), (x2, y2), green)

    cv2.imshow(win, vis)
    cv2.waitKey()

def crop_image_by_ratio(image, ratio):
    """
    crop image by ratio
    :param image: source image
    :param ratio: (top, bottom, left, right, )
    :return:crop image
    """
    (height, width) = image.shape
    up = int(height*ratio[0])
    down = int(height*ratio[1])
    left = int(width*ratio[2])
    right = int(width*ratio[3])
    return image[up:down,left:right]

def crop_rotate_rect(image, box, rect):
    """
    get image in rectangle and crop
    :param image: source image
    :param box: box from cv2.minAreaRect
    :param rect: rect from cv2.minAreaRect
    :return:crop image
    """
    W = rect[1][0]
    H = rect[1][1]
    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)
    angle = rect[2]
    if W<H and angle<-45:
        angle +=90
        croppedW = H if H > W else W
        croppedH = H if H < W else W
    elif W<H and angle>-45:
        croppedW = H if H < W else W
        croppedH = H if H > W else W
    elif W>H and angle<-45:
        angle += 90
        croppedW = H if H < W else W
        croppedH = H if H > W else W
    else:
        croppedW = H if H > W else W
        croppedH = H if H < W else W
    # Center of rectangle in source image
    center = ((x1 + x2) / 2 , (y1 + y2) / 2)
    # Size of the upright rectangle bounding the rotated rectangle
    size = (x2 - x1 , y2 - y1)
    M = cv2.getRotationMatrix2D((size[0] / 2 , size[1] / 2) , angle , 1.0)
    # Cropped upright rectangle
    cropped = cv2.getRectSubPix(image , size , center)
    cropped = cv2.warpAffine(cropped , M , size)
    # Final cropped & rotated rectangle
    croppedRotated = cv2.getRectSubPix(cropped , (int(croppedW) , int(croppedH)) , (size[0] / 2 , size[1] / 2))
    return croppedRotated

def put_text(image, text, position, font_size, color):
    """
    add text to image, the text can be chinese
    :param image:
    :param text:
    :param position:
    :param font_size:
    :param color:
    :return:image with text
    """
    pil_image = Image.fromarray(image)
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    font = ImageFont.truetype('../font/msyh.ttf', font_size)
    draw = ImageDraw.Draw(pil_image)
    draw.text(position, unicode(text, 'UTF-8'), font=font, fill=color)
    open_cv_image = np.array(pil_image)
    # Convert RGB to BGR
    # open_cv_image = open_cv_image[: , : , ::-1].copy()
    return open_cv_image

class TemplateMatcher():
    def __init__(self, _posive_templates_paths, _negive_templates_paths, _crop_w = (0, 1), _crop_h = (0, 1), _posive_threshold = 0.155, _negive_threshold = 0.14):
        self.update_templates(_posive_templates_paths, _negive_templates_paths)
        self.set_crop_area(_crop_w, _crop_h)
        self.set_threshold(_posive_threshold, _negive_threshold)
        return

    def set_threshold(self, _posive_threshold, _negive_threshold):
        self.__posive_threshold = _posive_threshold
        self.__negive_threshold = _negive_threshold
        return

    def set_crop_area(self, _crop_w, _crop_h):
        self.__crop_w = _crop_w
        self.__crop_h = _crop_h
        return

    def update_templates(self, _posive_templates_paths, _negive_templates_paths):
        self.__posive_templates = []
        for name in _posive_templates_paths:
            img = cv2.imread(name, 0)
            if img is None:
                print "no image in", name
            self.__posive_templates.append(img)

        self.__negive_templates = []
        for name in _negive_templates_paths:
            img = cv2.imread(name, 0)
            if img is None:
                raise "no image in", name
            self.__negive_templates.append(img)

        print len(self.__posive_templates), " posive templates recored"
        print len(self.__negive_templates), " negive templates recored"

    def match(self, _frame):
        cropped_frame = _frame[self.__crop_h[0] * _frame.shape[0]:self.__crop_h[1] * _frame.shape[0],
                       self.__crop_w[0] * _frame.shape[1]:self.__crop_w[1] * _frame.shape[1]]

        cropped_frame = cv2.GaussianBlur(cropped_frame, (3, 3), 200)
        # cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
        cropped_frame = cv2.equalizeHist(cropped_frame)
        cropped_frame = cv2.Canny(cropped_frame, 50, 230)

        posive_val, posive_loc, posive_w, posive_h = self.__template_math_serial(cropped_frame, self.__posive_templates)
        negive_val, negive_loc, negive_w, negive_h = self.__template_math_serial(cropped_frame, self.__negive_templates)

        # print "cas",posive_loc,negive_loc

        if posive_val > negive_val and posive_val > self.__posive_threshold:
            x = posive_loc[0] + _frame.shape[1]*self.__crop_w[0]
            y = posive_loc[1] + _frame.shape[0]*self.__crop_h[0]
            return 1, posive_val, (x, y), posive_w, posive_h
        if posive_val < negive_val and negive_val > self.__negive_threshold:
            x = negive_loc[0] + _frame.shape[1]*self.__crop_w[0]
            y = negive_loc[1] + _frame.shape[0]*self.__crop_h[0]
            return -1, negive_val, (x, y), negive_w, negive_h

        return 0, None, None, None, None

    def __template_math_serial(self, _frame, _templates):
        assert (len(_templates) > 0)
        score = 0
        location = None
        w = 0
        h = 0
        for template in _templates:
            maxVal, maxLoc = self.__template_match(_frame, template)
            if score < maxVal:
                score = maxVal
                location = maxLoc
                w = template.shape[1]
                h = template.shape[0]
        return score, location, w, h

    def __template_match(self, _frame, _template):
        template_match = cv2.matchTemplate(_frame, _template, cv2.TM_CCORR_NORMED)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(template_match)
        return maxVal, maxLoc

    ##TODO make the threshold trainble
    def train(self):
        return

class Detecter(object):
    """
    class that contains different kinds of detecters,
    """
    def __init__(self, detecter_type="HOG", classify_type="INVERSE"):
        """
        detecter_type:SIFT, MINUS, HIST, SOBEL_MINUS, CANNY_MINUS, CANNY_WINDOW, INVERSE_SIFT, CANNY_INVERSE, CANNY_P7003
        :param detecter_type:
        :param classify_type:
        """
        self.detecter_type = detecter_type
        self.template_gray = []
        self.classify_type = classify_type
        self.sift = None

    def set_template(self, template_image, template_name, set_inverse=True):
        """
        set template for detecter
        :param template_image: template image
        :param template_name: template name
        :param set_inverse: need to add inverse image?
        :return: None
        """
        template_image_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
        (width , height) = (template_image_gray.shape[1] , template_image_gray.shape[0])
        if width < height:
            template_image_gray = rotate_about_center(template_image_gray , 90)
        self.template_gray.append({"name":template_name, "image":template_image_gray})
        if set_inverse:
            template_image_gray_inverse = rotate_about_center(template_image_gray, 180)

            self.template_gray.append({"name": template_name , "image": template_image_gray_inverse})

    def clear_template(self):
        """
        delete templates
        :return: None
        """
        self.template_gray = []
        self.template_sobel_x = []

    def detect(self, template_image, image, **kwargs):
        """
        detect image
        :param template_image: template image
        :param image: image for detect, gray
        :param kwargs:
        :return: a list of score
        """
        if self.detecter_type == "MINUS":
            image = cv2.resize(image , (template_image.shape[1] , template_image.shape[0]))
            return np.mean(np.square(np.mean(template_image)-np.mean(image)))
        elif self.detecter_type == "HIST":
            image = cv2.resize(image , (template_image.shape[1] , template_image.shape[0]))
            template_image_hist = cv2.calcHist([template_image],[0],None,[256],[0.0,255.0]) 
            image_hist = cv2.calcHist([image],[0],None,[256],[0.0,255.0])
            degree = 0
            for i in range(len(template_image_hist)):
                if template_image_hist[i] != image_hist[i]:
                    degree = degree + (1 - abs(template_image_hist[i] - image_hist[i]) / max(template_image_hist[i] , image_hist[i]))
                else:
                    degree = degree + 1
            degree = degree / len(template_image_hist)
            return - degree[0]
        elif self.detecter_type == "SOBEL_MINUS":
            image = cv2.resize(image , (template_image.shape[1] , template_image.shape[0]))
            template_image_sobel = cv2.Sobel(template_image, -1,1,1)
            image_sobel = cv2.Sobel(image, -1, 1, 1)
            cv2.imshow("sobel" , template_image_sobel)
            cv2.imshow("sobel_image" , image_sobel)
            cv2.waitKey()
            return np.mean(np.square(np.mean(template_image_sobel)-np.mean(image_sobel)))
        elif self.detecter_type == "CANNY_MINUS":
            image = cv2.resize(image , (template_image.shape[1] , template_image.shape[0]))
            template_image_canny = canny_image(template_image, kernel=1)
            image_canny = canny_image(image, kernel=1)
            # cv2.imshow("sobel" , template_image_canny)
            # cv2.imshow("sobel_image" , image_canny)
            # cv2.waitKey()
            return np.mean(np.square(np.mean(template_image_canny) - np.mean(image_canny)))
        elif self.detecter_type == "CANNY_WINDOW":
            image = cv2.resize(image , (template_image.shape[1] , template_image.shape[0]))
            template_image_canny = canny_image(template_image , kernel=1)
            image_canny = canny_image(image , kernel=1)
            (height, width) = template_image_canny.shape
            window = (10, height)
            slide = (2, 2)
            template_image_deal = []
            for i in range(0, height-window[0]-1, slide[1]):
                temp = []
                for j in range(0, width-window[1]-1, slide[0]):
                    template_window = template_image_canny[i:i+window[1], j:j+window[0]]
                    # cv2.imshow("d", template_window)
                    # cv2.waitKey()
                    image_window = image_canny[i:i + window[1] , j:j + window[0]]
                    temp.append(np.mean(template_window)- np.mean(image_window))
                template_image_deal.append(temp)
            deal_width = len(template_image_deal)
            weights = np.array([np.exp(-np.square((i-deal_width/2.0)/deal_width)) for i in range(deal_width)])
            template_image_deal_square = [np.square(weights[i] * np.array(template_image_deal[i])) for i in range(deal_width)]
            return np.mean(template_image_deal_square)
        elif self.detecter_type == "SIFT":
            # import time
            # start = time.time()
            if self.sift is None:
                self.sift = surf = cv2.xfeatures2d.SIFT_create()

            kp1 , des1 = self.sift.detectAndCompute(template_image , None)
            kp2 , des2 = self.sift.detectAndCompute(image , None)

            # BFmatcher with default parms
            bf = cv2.BFMatcher(cv2.NORM_L2)
            matches = bf.knnMatch(des1 , des2 , k=2)

            p1 , p2 , kp_pairs = filter_matches(kp1 , kp2 , matches , ratio=0.8)
            if len(p1) > 0:
                H , status = cv2.findHomography(p1 , p2 , cv2.RANSAC , 5.0)
                #print H
                # print '%d / %d  inliers/matched' % (np.sum(status) , len(status))
                # do not draw outliers (there will be a lot of them)
                # kp_pairs = [kpp for kpp , flag in zip(kp_pairs , status) if flag]
            else:
                H , status = None , 0
                print '%d matches found, not enough for homography estimation' % len(p1)
            # end = time.time()
            # print "SIFT time:%f"%(end-start)
            # explore_match('matches' , template_image , image , kp_pairs , H=H)
            return -int(np.sum(status))
        elif self.detecter_type == "INVERSE_SIFT":
            dist1 = self.detect_with_type(template_image, template_image, "CANNY_INVERSE")
            dist2 = self.detect_with_type(template_image, image, "CANNY_INVERSE")
            if dist1 < 1000 and dist2 < 1000:
                self.detecter_type = "INVERSE_SIFT"
                return -1000
            else:
                result = self.detect_with_type(template_image, image, "SIFT")
                self.detecter_type = "INVERSE_SIFT"
                return result

        elif self.detecter_type == "CANNY_INVERSE":
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # image = cv2.resize(image , (template_image.shape[1] , template_image.shape[0]))
            ratio = (0.38, 0.66, 0.4, 0.6)
            # ratio = (0, 1, 0, 1)
            # template_image = crop_image_by_ratio(template_image, ratio)
            image = crop_image_by_ratio(image, ratio)
            # cv2.imshow("dd", template_image)
            # cv2.imshow("dd2", image)
            # cv2.waitKey(37)
            # template_image_canny = canny_image(template_image, kernel=15)
            image_canny = canny_image(image, kernel=1, bound=50)
            # cv2.imshow("template", template_image_canny)
            # cv2.imshow("image_canny", image_canny)
            # cv2.waitKey(37)
            dist = np.square(np.mean(image_canny))
            return dist
        elif self.detecter_type == "CANNY_P7003":
            image = cv2.resize(image , (template_image.shape[1] , template_image.shape[0]))
            # ratio = (0.38 , 0.66 , 0.4 , 0.6)
            ratio = (0, 1, 0, 1)
            template_image = crop_image_by_ratio(template_image , ratio)

            image = crop_image_by_ratio(image , ratio)
            # cv2.imshow("dd", template_image)
            # cv2.imshow("dd2" , image)
            # cv2.waitKey(37)
            template_image_canny = canny_image(template_image , kernel=3)
            image_canny = canny_image(image , kernel=3)
            # cv2.imshow("template", template_image_canny)
            # cv2.imshow("image_1" , image_canny)
            # cv2.waitKey()
            dist = np.mean(np.square(np.mean(template_image_canny) - np.mean(image_canny)))
            return dist
        elif self.detecter_type == "TEMPLATE_MATCH":
            template_match = cv2.matchTemplate(image, template_image, cv2.TM_CCORR_NORMED)
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(template_match)
            return -maxVal


    def detect_with_type(self, template_image, image, detecter_type):
        """
        detect with fix type
        :param template_image:
        :param image:
        :param detecter_type:
        :return:
        """
        self.detecter_type = detecter_type
        return self.detect(template_image, image)

    def classify(self, image, judge_bound, debug=False):
        """
        :param image:
        :return:
        """
        result_value = []
        template = self.template_gray
        for i in range(len(template)):
            result_value.append(self.detect(template[i]["image"], image))
            # cv2.imshow("image", image)
            # cv2.imshow("template", template[i]['image'])
            # print result
            # cv2.waitKey()
        # print np.array(result)
        min_arg = np.argmin(np.array(result_value))
        return self.get_result(result_value, min_arg, judge_bound, debug=debug)

    def get_result(self, result_value, min_arg, judge_bound, debug=False):
        if self.classify_type == "INVERSE":
            value = result_value[min_arg]
            if debug:
                print "value:" + str(value)
            if value < judge_bound:
                return "反面"
            else:
                return "正面"
        elif self.classify_type == "P7003":
            value = result_value[min_arg]
            if debug:
                print "value:" + str(value)
            if value < judge_bound:
                return "P7003"
            else:
                return "other"
        elif self.classify_type == "ALL":
            name = self.template_gray[min_arg]['name']
            return name

    def end(self):
        print "end"

import multiprocessing
class Consumer(multiprocessing.Process, Detecter):
    def __init__(self, task_q, result_q, detecter_type):
        print "initialize"
        multiprocessing.Process.__init__(self)
        self.task_q = task_q
        self.result_q = result_q
        self.detecter_type = detecter_type

    def run(self):
        Detecter.__init__(self, self.detecter_type)
        while True:
            next_task = self.task_q.get(block=True)
            if next_task[0] == 2:
                print "kill"
                self.task_q.task_done()
            elif next_task[0] == 1:
                template_image = next_task[1]
                image = next_task[2]
                task_id = next_task[3]
                result = Detecter.detect(self, template_image, image)
                self.result_q.put((task_id, result))
                self.task_q.task_done()
        return


class DetecterMultiProcess(Detecter):
    def __init__(self, jobs=2, detecter_type="CANNY_MINUS", classify_type="INVERSE"):
        Detecter.__init__(self, detecter_type=detecter_type)
        self.tasks = multiprocessing.JoinableQueue()
        self.results = multiprocessing.Queue()
        self.jobs = jobs
        self.consumers = []
        self.classify_type = classify_type
        for i in xrange(jobs):
            self.consumers.append(Consumer(self.tasks, self.results, detecter_type))
        for a in self.consumers:
            a.start()

    def classify(self, image, judge_bound, debug=False):
        """
            :param image:
            :return:
            """
        import time
        # start = time.time()

        result = []
        result_value = []
        template = self.template_gray
        for i in range(len(template)):
            # image = cv2.resize(image , (template[i]['image'].shape[1] , template[i]['image'].shape[0]))
            self.tasks.put((1, template[i]['image'], image, i))
        self.tasks.join()
        for i in range(len(template)):
            get = self.results.get()
            result.append(get)
            result_value.append((get[1]))
        min_arg = np.argmin(np.array(result_value))
        min_index = result[min_arg][0]

        # end = time.time()
        # print "classify time:"+str(end-start)
        return self.get_result(result_value, min_index, judge_bound, debug=debug)

    def end(self):
        for i in xrange(self.jobs):
            self.tasks.put((2, 2, 2))