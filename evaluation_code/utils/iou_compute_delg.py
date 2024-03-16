import os
import argparse
import json

import cv2
import numpy as np


def extract_bounding_box(image, points):
    points = np.array(points)
    x_min = points[:, 0].min()
    x_max = points[:, 0].max()
    y_min = points[:, 1].min()
    y_max = points[:, 1].max()

    bbox = cv2.rectangle(image.copy(), (int(x_min), int(y_min)), (int(x_max) + 1, int(y_max) + 1), (255, 0, 0), 5)

    bbox_masked = image.copy()
    height, width, _ = bbox_masked.shape
    for w in range(width):
        for h in range(height):
            if w < x_min or w > x_max:
                bbox_masked[h, w, :] = [0, 0, 0]
            if h < y_min or h > y_max:
                bbox_masked[h, w, :] = [0, 0, 0]

    return bbox, bbox_masked


def extract_segmentation(image, points):
    points = np.array(points)
    points = points.astype(int)

    x = points[:, 0]
    y = points[:, 1]

    contour = np.array([[xii, yii] for xii, yii in zip(x.astype(int), y.astype(int))])
    contour_mask = contour.copy().reshape((1, -1, 2))
    segmentation = cv2.polylines(image.copy(), [contour_mask], True, (255, 0, 0), 5)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, pts=[contour], color=(255, 255, 255))
    segmentation_masked = cv2.bitwise_and(image, mask)

    return segmentation, segmentation_masked


''' type = bbox or segmentation '''
def compute_iou(test_points, gt_points, image=None, type='bbox'):
    " if image!=None, draw the result "
    
    #combine the points
    all_gt_points=[]
    for i in range(len(gt_points)):
        all_gt_points=all_gt_points+gt_points[i]

    test_points, all_gt_points = np.array(test_points).astype(int), np.array(all_gt_points).astype(int)
    image_size = max(test_points[:, 0].max(),test_points[:, 1].max(),all_gt_points[:, 0].max(),all_gt_points[:, 1].max())
    test_mask, gt_mask = np.zeros((image_size, image_size)), np.zeros((image_size, image_size))

    if type == 'bbox':

        # test_x_min = test_points[:, 0].min()
        # test_x_max = test_points[:, 0].max()
        # test_y_min = test_points[:, 1].min()
        # test_y_max = test_points[:, 1].max()

        test_x_min = test_points[:, 1].min()
        test_x_max = test_points[:, 1].max()
        test_y_min = test_points[:, 0].min()
        test_y_max = test_points[:, 0].max()
        test_mask[test_y_min:test_y_max+1, test_x_min:test_x_max+1] = 1.

        gt_x_min = all_gt_points[:, 0].min()
        gt_x_max = all_gt_points[:, 0].max()
        gt_y_min = all_gt_points[:, 1].min()
        gt_y_max = all_gt_points[:, 1].max()
        gt_mask[gt_y_min:gt_y_max+1, gt_x_min:gt_x_max+1] = 1.

        intersection = np.sum(np.logical_and(test_mask, gt_mask))
        union = np.sum(gt_mask) + np.sum(test_mask) - intersection
        iou = intersection / union

        if image is None:
            return iou
        else:
            bbox = cv2.rectangle(image.copy(), (int(gt_x_min), int(gt_y_min)),
                                    (int(gt_x_max) + 1, int(gt_y_max) + 1), (0, 0, 255), 5)
            bbox = cv2.rectangle(bbox, (int(test_x_min), int(test_y_min)),
                                    (int(test_x_max) + 1, int(test_y_max) + 1), (255, 0, 0), 5)

        return iou, bbox

    elif type == 'segmentation':

        
        test_x = test_points[:, 1]
        test_y = test_points[:, 0]
                
        # test_x = test_points[:, 0]
        # test_y = test_points[:, 1]

        test_contour = np.array([[xii, yii] for xii, yii in zip(test_x.astype(int), test_y.astype(int))])
        cv2.fillPoly(test_mask, pts=[test_contour], color=(255, 255, 255))

        test_mask /= 255.

        for i in range(len(gt_points)):
            points=gt_points[i]
            points=np.array(points).astype(int)
            gt_x = points[:, 0]
            gt_y = points[:, 1]
            gt_contour = np.array([[xii, yii] for xii, yii in zip(gt_x.astype(int), gt_y.astype(int))])
            cv2.fillPoly(gt_mask, pts=[gt_contour], color=(255, 255, 255))
        gt_mask /= 255.

        intersection = np.sum(np.logical_and(test_mask, gt_mask))
        union = np.sum(gt_mask) + np.sum(test_mask) - intersection
        iou = intersection / union

        if image is None:
            return iou
        else:
            test_contour_mask = test_contour.copy().reshape((1, -1, 2))
            gt_contour_mask = gt_contour.copy().reshape((1, -1, 2))
            segmentation = cv2.polylines(image.copy(), [gt_contour_mask], True, (0, 0, 255), 5)
            segmentation = cv2.polylines(segmentation, [test_contour_mask], True, (255, 0, 0), 5)

            return iou, segmentation
