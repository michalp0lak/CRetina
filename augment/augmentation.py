import cv2
import numpy as np
import random
from augment.ops import matrix_iof 
from matplotlib import pyplot as plt

class Augmentation():

    """Class consisting common augmentation methods for image object detection pipeline."""

    def __init__(self, cfg, seed=None):

        self.img_dim = cfg['image_size']
        self.rgb_means = cfg['rgb_means']

    def crop(self, image, boxes, labels, img_dim):

        height, width, _ = image.shape
        pad_image_flag = True

        # Crop procedure performs 250 trials to crop image, if not succeed it returns original image
        for _ in range(250):
            """
            if random.uniform(0, 1) <= 0.2:
                scale = 1.0
            else:
                scale = random.uniform(0.3, 1.0)
            """
            # Randomly select scale
            PRE_SCALES = [0.3, 0.45, 0.6, 0.8, 1.0]
            scale = random.choice(PRE_SCALES)
            # Define square shape of image
            short_side = min(width, height)
            w = int(scale * short_side)
            h = w

            # Define ROI pixels
            if width == w:
                l = 0
            else:
                l = random.randrange(width - w)
            if height == h:
                t = 0
            else:
                t = random.randrange(height - h)
            roi = np.array((l, t, l + w, t + h))

            # If any bbox covers whole ROI -> invalid cropping
            value = matrix_iof(boxes, roi[np.newaxis])
            flag = (value >= 1)
            if not flag.any():
                continue
            
            # Check for bboxes located in ROI
            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            mask_a = np.logical_and(roi[:2] < centers, centers < roi[2:]).all(axis=1)
            boxes_t = boxes[mask_a].copy()
            labels_t = labels[mask_a].copy()

            if boxes_t.shape[0] == 0:
                continue
            
            # Crop image and bboxes
            image_t = image[roi[1]:roi[3], roi[0]:roi[2]]

            boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
            boxes_t[:, :2] -= roi[:2]
            boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
            boxes_t[:, 2:] -= roi[:2]

	        # make sure that the cropped image contains at least one bbox > 16 pixel at training image scale
            b_w_t = (boxes_t[:, 2] - boxes_t[:, 0] + 1) / w * img_dim
            b_h_t = (boxes_t[:, 3] - boxes_t[:, 1] + 1) / h * img_dim
            mask_b = np.minimum(b_w_t, b_h_t) > 0.0
            boxes_t = boxes_t[mask_b]
            labels_t = labels_t[mask_b]

            if boxes_t.shape[0] == 0:
                continue

            pad_image_flag = False

            return image_t, boxes_t, labels_t, pad_image_flag
        return image, boxes, labels, pad_image_flag

    def distort(self, image):

        def apply_distortion(image, alpha=1, beta=0):

            tmp = image.astype(float) * alpha + beta
            tmp[tmp < 0] = 0
            tmp[tmp > 255] = 255
            image[:] = tmp

        image = image.copy()

        if random.randrange(2):

            #brightness distortion
            if random.randrange(2):
                apply_distortion(image, beta=random.uniform(-32, 32))

            #contrast distortion
            if random.randrange(2):
                apply_distortion(image, alpha=random.uniform(0.5, 1.5))
        
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            #saturation distortion
            if random.randrange(2):
                apply_distortion(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

            #hue distortion
            if random.randrange(2):
                tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
                tmp %= 180
                image[:, :, 0] = tmp

            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        else:

            #brightness distortion
            if random.randrange(2):
                apply_distortion(image, beta=random.uniform(-32, 32))

            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            #saturation distortion
            if random.randrange(2):
                apply_distortion(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

            #hue distortion
            if random.randrange(2):
                tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
                tmp %= 180
                image[:, :, 0] = tmp

            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

            #contrast distortion
            if random.randrange(2):
                apply_distortion(image, alpha=random.uniform(0.5, 1.5))

        return image

    def expand(self, image, boxes, fill, p):

        # Randomly choose if to expand
        if random.randrange(2):
            return image, boxes

        height, width, depth = image.shape

        # Get expanded dimensions
        scale = random.uniform(1, p)
        w = int(scale * width)
        h = int(scale * height)

        # Get offsets
        left = random.randint(0, w - width)
        top = random.randint(0, h - height)

        # Shift the boxes with offsets
        boxes_t = boxes.copy()
        boxes_t[:, :2] += (left, top)
        boxes_t[:, 2:] += (left, top)

        # Empty expanded image
        expand_image = np.empty(
            (h, w, depth),
            dtype=image.dtype)
            
        # Fill with triplet/color
        expand_image[:, :] = fill
        # Fit original into expanded canvas
        expand_image[top:top + height, left:left + width] = image

        return expand_image, boxes_t

    def mirror(self, image, boxes):

        _, width, _ = image.shape
        
        if random.randrange(2):

            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]

        return image, boxes

    def pad_to_square(self, image, rgb_mean, pad_image_flag):

        if not pad_image_flag:
            return image

        height, width, _ = image.shape
        long_side = max(width, height)
        image_t = np.empty((long_side, long_side, 3), dtype=image.dtype)
        image_t[:, :] = rgb_mean
        image_t[0:0 + height, 0:0 + width] = image

        return image_t

    def resize_subtract_mean(self, image, insize, rgb_mean):

        interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        #interp_method = interp_methods[random.randrange(5)]
        interp_method = interp_methods[0]    
 
        image = cv2.resize(image, (insize, insize), interpolation=interp_method)
        image = image.astype(np.float32)
        #image = (image-rgb_mean)
        image /= 255

        return image

    def augment(self, data, split):

        image = data['image'].copy()
        boxes = data['boxes'].copy()
        labels = data['labels'].copy()
        
        assert boxes.shape[0] > 0, "this image does not contain any annotation"

        image_t = image.copy()
        boxes_t = boxes.copy()
        labels_t = labels.copy()
        
        # Crop
        #if split in ['training']:
        #    image_t, boxes_t, labels_t, pad_image_flag = self.crop(image, boxes, labels, self.img_dim)
        #Distort
        #if split in ['training']:
        #    image_t = self.distort(image_t)
        # Pad to square if not square
        #if split in ['training', 'validation']:
        #    image_t = self.pad_to_square(image_t, self.rgb_means, pad_image_flag=True)
        # Mirror
        #if split in ['training']:
        #    image_t, boxes_t = self.mirror(image_t, boxes_t)
        # Get dims for bboxes coords calculation
        height, width, _ = image_t.shape
        # Resize image to required shape and substract mean
        if split in ['training', 'validation']:
            image_t = self.resize_subtract_mean(image_t, self.img_dim, self.rgb_means)

        # Get relative location of bboxes
        boxes_t[:, 0::2] /= width
        boxes_t[:, 1::2] /= height
        data['centers'][0] /= width
        data['centers'][1] /= height
        labels_t = np.expand_dims(labels_t, 1)

        return {'image': image_t, 'labels':labels_t, 'boxes': boxes_t}