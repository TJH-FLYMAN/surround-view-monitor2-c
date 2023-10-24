import cv2
import numpy as np
import os 
camera_names = ["front", "back", "left", "right"]

# --------------------------------------------------------------------
# (shift_width, shift_height): how far away the birdview looks outside
# of the calibration pattern in horizontal and vertical directions
shift_w = 300
shift_h = 300

# size of the gap between the calibration pattern and the car
# in horizontal and vertical directions
inn_shift_w = 20
inn_shift_h = 50

# total width/height of the stitched image
total_w = 600 + 2 * shift_w
total_h = 1000 + 2 * shift_h

# four corners of the rectangular region occupied by the car
# top-left (x_left, y_top), bottom-right (x_right, y_bottom)
xl = shift_w + 180 + inn_shift_w
xr = total_w - xl
yt = shift_h + 200 + inn_shift_h
yb = total_h - yt
# --------------------------------------------------------------------

project_shapes = {
    "front": (total_w, yt),
    "back":  (total_w, yt),
    "left":  (total_h, xl),
    "right": (total_h, xl)
}
project_keypoints = {
    "front": [(shift_w + 120, shift_h),
              (shift_w + 480, shift_h),
              (shift_w + 120, shift_h + 160),
              (shift_w + 480, shift_h + 160)],

    "back":  [(shift_w + 120, shift_h),
              (shift_w + 480, shift_h),
              (shift_w + 120, shift_h + 160),
              (shift_w + 480, shift_h + 160)],

    "left":  [(shift_h + 280, shift_w),
              (shift_h + 840, shift_w),
              (shift_h + 280, shift_w + 160),
              (shift_h + 840, shift_w + 160)],

    "right": [(shift_h + 160, shift_w),
              (shift_h + 720, shift_w),
              (shift_h + 160, shift_w + 160),
              (shift_h + 720, shift_w + 160)]
}
def FI(front_image):
    return front_image[:, :xl]
def FII(front_image):
    return front_image[:, xr:]
def FM(front_image):
    return front_image[:, xl:xr]
def BIII(back_image):
    return back_image[:, :xl]
def BIV(back_image):
    return back_image[:, xr:]
def BM(back_image):
    return back_image[:, xl:xr]
def LI(left_image):
    return left_image[:yt, :]
def LIII(left_image):
    return left_image[yb:, :]
def LM(left_image):
    return left_image[yt:yb, :]
def RII(right_image):
    return right_image[:yt, :]
def RIV(right_image):
    return right_image[yb:, :]
def RM(right_image):
    return right_image[yt:yb, :]
def get_mask(img):
    """
    Convert an image to a mask array.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    return mask


def get_overlap_region_mask(imA, imB):
    """
    Given two images of the save size, get their overlapping region and
    convert this region to a mask array.
    """
    overlap = cv2.bitwise_and(imA, imB)
    mask = get_mask(overlap)
    mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=2)
    return mask


def get_outmost_polygon_boundary(img):
    """
    Given a mask image with the mask describes the overlapping region of
    two images, get the outmost contour of this region.
    """
    mask = get_mask(img)
    mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=2)
    cnts, hierarchy = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)[-2:]

    # get the contour with largest aera
    C = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)[0]

    # polygon approximation
    polygon = cv2.approxPolyDP(C, 0.009 * cv2.arcLength(C, True), True)

    return polygon
i= 0
Note=open('/home/tjh/hello/hello-test/images/result_python.txt',mode='w')
def get_weight_mask_matrix(imA, imB, dist_threshold=5):
    """
    Get the weight matrix G that combines two images imA, imB smoothly.
    """
    # i= 0
    # Note=open('/home/tjh/hello/hello-test/images/result_python.txt',mode='w')
    overlapMask = get_overlap_region_mask(imA, imB)
    overlapMaskInv = cv2.bitwise_not(overlapMask)
    indices = np.where(overlapMask == 255)
    imA_diff = cv2.bitwise_and(imA, imA, mask=overlapMaskInv)
    imB_diff = cv2.bitwise_and(imB, imB, mask=overlapMaskInv)
    G = get_mask(imA).astype(np.float32) / 255.0
    polyA = get_outmost_polygon_boundary(imA_diff)
    polyB = get_outmost_polygon_boundary(imB_diff)
    for y, x in zip(*indices):

        #convert this x,y int an INT tuple
        xy_tuple = tuple([int(x), int(y)])
        distToB = cv2.pointPolygonTest(polyB, xy_tuple, True)
        if distToB < dist_threshold:
            distToA = cv2.pointPolygonTest(polyA, xy_tuple, True)
            distToB *= distToB
            distToA *= distToA
            dist = distToB / (distToA + distToB)
            #Note.write("%s\n" % (dist * 255.0))
            print( x,"|",y, "|",dist*255.0)
            G[y, x] = dist
    #Note.close()
    cv2.imshow('G',G)
    cv2.waitKey(0)
    return G, overlapMask

front=cv2.imread("/home/tjh/hello/hello-test/images/front_rotate.png")

back=cv2.imread("/home/tjh/hello/hello-test/images/back_rotate.png")
left=cv2.imread("/home/tjh/hello/hello-test/images/left_rotate.png")
right = cv2.imread("/home/tjh/hello/hello-test/images/right_rotate.png")
G0, M0 = get_weight_mask_matrix(FI(front), LI(left))
print(G0[55,55])
# cv2.imwrite("/home/tjh/hello/hello-test/images/python/G0.jpg",G0*255)
# cv2.imwrite("/home/tjh/hello/hello-test/images/python/M0.png",M0)
# cv2.imshow("G0",G0)
# cv2.waitKey(0)
# print(G0.shape,G0.size,G0.ndim,G0.dtype)
# print(front.shape,back.shape,left.shape,right.shape)
# G1, M1 = get_weight_mask_matrix(FII(front), RII(right))
# G2, M2 = get_weight_mask_matrix(BIII(back), LIII(left))
# G3, M3 = get_weight_mask_matrix(BIV(back), RIV(right))
# self.weights = [np.stack((G, G, G), axis=2) for G in (G0, G1, G2, G3)]