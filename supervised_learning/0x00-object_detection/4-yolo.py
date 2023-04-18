#!/usr/bin/env python3
"""Task 4. Load_images added to 3-yolo.py"""
from tensorflow.keras.models import load_model
import numpy as np
import glob
import cv2


class Yolo:
    """Uses You only look once (YOLO)v3 to perform object detection
    Args:
        model_path: path to where a Darknet Keras model is stored
        classes_path: path to where the list of class names used for the
            Darknet model, listed in order of index, can be found
        class_t: float representing the box score threshold for the
            initial filtering step
        nms_t: float representing the IOU threshold for non-max
            suppression
        anchors: numpy.ndarray of shape (outputs, anchor_boxes, 2)
            containing all of the anchor boxes:
            outputs: is the number of outputs (predictions) made by the
                Darknet model
            anchor_boxes: is the number of anchor boxes used for each
                prediction
            2 => [anchor_box_width, anchor_box_height]
    """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        self.model = load_model(model_path)
        with open(classes_path) as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """Process Darknet model outputs
        Args:
            outputs: list of numpy.ndarrays containing the predictions from
                the Darknet model for a single image:
                Each output will have the shape (grid_height, grid_width,
                    anchor_boxes, 4 + 1 + classes)
                    grid_height & grid_width => the height and width of the
                        grid used for the output
                    anchor_boxes => the number of anchor boxes used
                    4 => (t_x, t_y, t_w, t_h)
                    1 => box_confidence
                    classes => class probabilities for all classes
            image_size: numpy.ndarray containing the image’s original size
                [image_height, image_width]
        Returns:
            A tuple of (boxes, box_confidences, box_class_probs):
                boxes: a list of numpy.ndarrays of shape (grid_height,
                    grid_width, anchor_boxes, 4) containing the processed
                    boundary boxes for each output, respectively:
                    4 => (x1, y1, x2, y2)
                        (x1, y1, x2, y2) should represent the boundary box
                        relative to original image
                box_confidences: list of numpy.ndarrays of shape (grid_height,
                    grid_width, anchor_boxes, 1) containing the box confidences
                    for each output, respectively
                box_class_probs: list of numpy.ndarrays of shape (grid_height,
                    grid_width, anchor_boxes, classes) containing the box’s
                    class probabilities for each output, respectively
        """
        boxes = []
        box_confidences = []
        box_class_probs = []
        for i in range(len(outputs)):
            boxes.append(outputs[i][..., :4])
            box_confidences.append(1 / (1 + np.exp(-outputs[i][..., 4:5])))
            box_class_probs.append(1 / (1 + np.exp(-outputs[i][..., 5:])))
        image_height, image_width = image_size
        for i in range(len(boxes)):
            grid_width = outputs[i].shape[1]
            grid_height = outputs[i].shape[0]
            anchor_boxes = outputs[i].shape[2]
            for cy in range(grid_height):
                for cx in range(grid_width):
                    for b in range(anchor_boxes):
                        tx, ty, tw, th = boxes[i][cy, cx, b]
                        pw, ph = self.anchors[i][b]
                        bx = (1 / (1 + np.exp(-tx))) + cx
                        by = (1 / (1 + np.exp(-ty))) + cy
                        bw = pw * np.exp(tw)
                        bh = ph * np.exp(th)
                        bx /= grid_width
                        by /= grid_height
                        bw /= self.model.input.shape[1].value
                        bh /= self.model.input.shape[2].value
                        x1 = (bx - (bw / 2)) * image_width
                        y1 = (by - (bh / 2)) * image_height
                        x2 = (bx + (bw / 2)) * image_width
                        y2 = (by + (bh / 2)) * image_height
                        boxes[i][cy, cx, b] = [x1, y1, x2, y2]
        return (boxes, box_confidences, box_class_probs)

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Removes the boxes with low box scores
        Args:
            boxes: list of numpy.ndarrays of shape (grid_height, grid_width,
                anchor_boxes, 4) containing the processed boundary boxes for
                each output, respectively
            box_confidences: list of numpy.ndarrays of shape (grid_height,
                grid_width, anchor_boxes, 1) containing the processed box
                confidences for each output, respectively
            box_class_probs: list of numpy.ndarrays of shape (grid_height,
                grid_width, anchor_boxes, classes) containing the processed
                box class probabilities for each output, respectively
        Returns:
            A tuple of (filtered_boxes, box_classes, box_scores):
                filtered_boxes: a numpy.ndarray of shape (?, 4) containing all
                    of the filtered bounding boxes:
                box_classes: a numpy.ndarray of shape (?,) containing the class
                    number that each box in filtered_boxes predicts
                box_scores: a numpy.ndarray of shape (?) containing the box
                    scores"""
        filtered_boxes, box_classes, box_scores = None, [], []
        for i in range(len(boxes)):
            cur_box_score = box_confidences[i] * box_class_probs[i]
            cur_box_class = np.argmax(cur_box_score, axis=-1)
            cur_box_score = np.max(cur_box_score, axis=-1)
            mask = cur_box_score >= self.class_t
            if filtered_boxes is None:
                filtered_boxes = boxes[i][mask]
                box_scores = cur_box_score[mask]
                box_classes = cur_box_class[mask]
            else:
                filtered_boxes = np.concatenate((filtered_boxes,
                                                boxes[i][mask]),
                                                axis=0)
                box_classes = np.concatenate((box_classes,
                                              cur_box_class[mask]),
                                             axis=0)
                box_scores = np.concatenate((box_scores,
                                             cur_box_score[mask]),
                                            axis=0)
        return (filtered_boxes, box_classes, box_scores)

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Performs non-max suppression"""
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        for c in set(box_classes):
            idxs = np.where(box_classes == c)
            class_boxes = filtered_boxes[idxs]
            class_box_scores = box_scores[idxs]

            while len(class_boxes) > 0:
                max_idx = np.argmax(class_box_scores)
                box_predictions.append(class_boxes[max_idx])
                predicted_box_classes.append(c)
                predicted_box_scores.append(class_box_scores[max_idx])

                class_boxes = np.delete(class_boxes, max_idx, axis=0)
                class_box_scores = np.delete(class_box_scores, max_idx, axis=0)

                if len(class_boxes) == 0:
                    break

                iou = self.intersection_over_union(box_predictions[-1],
                                                   class_boxes)
                iou_mask = iou < self.nms_t

                class_boxes = class_boxes[iou_mask]
                class_box_scores = class_box_scores[iou_mask]

        return (np.array(box_predictions),
                np.array(predicted_box_classes),
                np.array(predicted_box_scores))

    def intersection_over_union(self, box1, boxes):
        """Calculates the intersection over union of two sets of boxes"""
        x1 = np.maximum(box1[0], boxes[:, 0])
        y1 = np.maximum(box1[1], boxes[:, 1])
        x2 = np.minimum(box1[2], boxes[:, 2])
        y2 = np.minimum(box1[3], boxes[:, 3])

        intersection_area = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        union_area = box1_area + boxes_area - intersection_area

        return intersection_area / union_area

    @staticmethod
    def load_images(folder_path):
        """Loads images from a filepath
        Args:
            folder_path: a string representing the path to the folder holding
                all the images to load
        Returns:
            a tuple of (images, image_paths):
                images: a list of images as numpy.ndarrays
                image_paths: a list of paths to the individual images in images
        """
        image_paths = glob.glob(folder_path + '/*')
        images = [cv2.imread(image) for image in image_paths]
        return images, image_paths
