import numpy as np
import cv2
import random

from ops import iou_jit


class OvalSimulator():

    def __init__(self, rng, oval_sim_cfg):

        self.rng = rng
        self.radius_limit = oval_sim_cfg['radius_limit']
        self.circle_prob = oval_sim_cfg['circle_prob']
        self.ellipse_aspect_ratio = oval_sim_cfg['ellipse_aspect_ratio']

        self.line_thickness = oval_sim_cfg['line_thickness']

        self.occlusion_prob = oval_sim_cfg['occlusion']['prob']
        self.occlusion_range = oval_sim_cfg['occlusion']['range']

    def circle_center_limits(self,radius, width, height):

        assert width//2 > radius, 'Circle does not fit in image'
        assert height//2 > radius, 'Circle does not fit in image'
        
        return (radius, width-radius), (radius, height-radius)

  
    def ellipse_center_limits(self,axesLength, angle, width, height):

        assert width//2 > axesLength[0], 'Ellipse does not fit in image'
        assert height//2 > axesLength[1], 'Ellipse does not fit in image'

        if angle == 0:

            return (axesLength[0], height - axesLength[0]), (axesLength[1], width - axesLength[1])

        else:

            Sin = np.sin(np.deg2rad(angle))
            Cos = np.cos(np.deg2rad(angle))

            tx = np.arctan(-(axesLength[1]*Sin)/(axesLength[0]*Cos))
            ty = np.arctan(-(axesLength[1]*Cos)/(axesLength[0]*Sin))

            x1 = axesLength[0]*Cos*np.cos(tx) - axesLength[1]*Sin*np.sin(ty)
            y1 = axesLength[0]*Sin*np.cos(tx) - axesLength[1]*Cos*np.sin(ty)
            x2 = axesLength[0]*Cos*np.cos(tx + np.pi) - axesLength[1]*Sin*np.sin(ty + np.pi)
            y2 = axesLength[0]*Sin*np.cos(tx + np.pi) - axesLength[1]*Cos*np.sin(ty + np.pi)

            x_low = np.round(np.abs(np.min([x1,x2])))
            x_up = width-np.round(np.max([x1,x2]))

            y_low = np.round(np.abs(np.min([y1,y2])))
            y_up = height-np.round(np.max([y1,y2]))

            return (int(x_low), int(x_up)), (int(y_low), int(y_up))

    def occlusion(self):

        if self.rng.binomial(n=1, p = self.occlusion_prob):

            # Randomly generate occlusion proportion with beta distribution
            occlusion = self.rng.beta(2,1.5)*(self.occlusion_range[1]-self.occlusion_range[0]) + self.occlusion_range[0]
            # Get start and end point of occlusion
            opening_angle = int(np.rad2deg(self.rng.integers(low=0, high=2*np.pi, size=1, endpoint = False)[0]))
            closing_angle = int(np.rad2deg(2*np.pi*(1-occlusion))) + opening_angle

            direction = ((opening_angle+closing_angle)/2)%360

            return opening_angle, closing_angle, direction

        else:

            return 0,360,0

    def generate_oval(self, width, height):

        # Circle = 1 X Ellipse = 0
        oval_shape = self.rng.binomial(n=1, p = self.circle_prob)

        # Radius
        radius = self.rng.integers(low=self.radius_limit[0], high=self.radius_limit[1], size=1, endpoint = True)[0]
        
        line_t = int(self.rng.integers(self.line_thickness[0], self.line_thickness[1], size = 1, endpoint = True)[0])

        # Generate circle attributes
        if oval_shape:

            x_center_limit, y_center_limit = self.circle_center_limits(radius+line_t, width, height)
            cx = self.rng.integers(low=x_center_limit[0], high=x_center_limit[1], size=1, endpoint = True)[0]
            cy = self.rng.integers(low=y_center_limit[0], high=y_center_limit[1], size=1, endpoint = True)[0]
            center_coordinates = [int(cx), int(cy)] 
            axesLength = [int(radius), int(radius)]
            angle = 0

        # Generate ellipse attributes
        else:

            aspect_ratio = self.rng.uniform(self.ellipse_aspect_ratio[0], self.ellipse_aspect_ratio[1], 1)
            axesLength = [int(radius), int(np.ceil(radius*aspect_ratio).astype(np.int32)[0])]
            
            angle = self.rng.uniform(0, np.pi, 1)[0]

            x_center_limit, y_center_limit = self.ellipse_center_limits(list(map(lambda x:x+line_t, axesLength)), angle, width, height)
            cx = self.rng.integers(low=x_center_limit[0], high=x_center_limit[1], size=1, endpoint = True)[0]
            cy = self.rng.integers(low=y_center_limit[0], high=y_center_limit[1], size=1, endpoint = True)[0]
            center_coordinates = [int(cx), int(cy)] 

        # Get bounding box
        oval_start, oval_end, orientation = self.occlusion()

        polygon = cv2.ellipse2Poly(center_coordinates, axesLength, int(np.rad2deg(angle)), 0, 360, 1)
        bbox = (np.min(polygon, axis = 0)-line_t).tolist() + (np.max(polygon, axis = 0)+line_t).tolist()

        polygon = cv2.ellipse2Poly(center_coordinates, axesLength, int(np.rad2deg(angle)), oval_start, oval_end, 1)
        label_bbox = (np.min(polygon, axis = 0)-line_t).tolist() + (np.max(polygon, axis = 0)+line_t).tolist()

        orientation += angle
  
        return {'class': oval_shape,'center': center_coordinates, 'axes':axesLength, 'angle': angle, 'bbox': bbox,
                'label_bbox': label_bbox, 'occlusion': [oval_start, oval_end], 'orientation': orientation,'line': line_t}

    def generate_ovals(self, image, item_num):
        
        h, w = image.shape[:2]
        sample = []
        current_boxes = []

        for i in range(item_num):

            if len(current_boxes) > 0:

                oval = self.generate_oval(width=w, height=h)
                iou = iou_jit(np.array(current_boxes).reshape(-1,4), np.array(oval['bbox']).reshape(-1,4))
    
                if np.all(iou == 0):
                    current_boxes.append(oval['bbox'])
                    sample.append(oval)

            else:
                oval = self.generate_oval(width=w, height=h)
                current_boxes.append(oval['bbox'])
                sample.append(oval)

        return sample

    
class LinearNoiseSimulator():

    def __init__(self, rng, line_range, corner_prob):

        self.rng = rng
        self.line_range = line_range
        self.corner_prob = corner_prob

    def valid_line_end(self, start_pixel, end_pixels, mask):

        endpoint_mask = np.zeros(shape=(end_pixels.shape[0],))
        
        for i in range(end_pixels.shape[0]):

            trial_mask = np.zeros(shape=mask.shape)
            trial_mask = cv2.line(trial_mask, (start_pixel[0], start_pixel[1]), 
                                (end_pixels[i][0], end_pixels[i][1]),
                                color = (255), thickness=2)

            endpoint_mask[i] = np.sum(np.multiply(trial_mask,mask)) == 0

        return np.where(endpoint_mask > 0)[0].tolist()

    def line_end(self, start_pixel, end_pixels, mask, avail_dirs):

        detected_valid_end = False
        endpixel_dir = None

        while (not detected_valid_end) and avail_dirs:

            sel_dir = random.choice(avail_dirs) 
            avail_dirs.remove(sel_dir)
            
            trial_mask = np.zeros(shape=mask.shape)

            trial_mask = cv2.line(trial_mask, (start_pixel[0], start_pixel[1]), 
                                (end_pixels[sel_dir][0], end_pixels[sel_dir][1]),
                                color = (255), thickness=2)
            
            if np.sum(np.multiply(trial_mask, mask)) == 0:
                detected_valid_end = True
                endpixel_dir = sel_dir

        return endpixel_dir, avail_dirs

    def generate_linear_noise(self, image, bbox_mask):

        # Find potential line start pixel
        available_pixels = np.concatenate((np.where(bbox_mask == 0)[1].reshape(-1,1), np.where(bbox_mask == 0)[0].reshape(-1,1)), axis=-1)
        start_pixel = random.choice(available_pixels)

        # Get line length and adjust it according to image size
        line_length = self.rng.integers(self.line_range[0], self.line_range[1], size=1, endpoint = True)[0]
        line_length = np.min([line_length, np.min(bbox_mask.shape)])

        # Set of potential end pixels
        end_pixels = cv2.ellipse2Poly(start_pixel.tolist(), (line_length,line_length), 0, 0, 360, 1)
        # Available end pixels directions
        endpixel_avail_dirs = np.arange(0,end_pixels.shape[0]).tolist()
        endpixel_dir, endpixel_avail_dirs = self.line_end(start_pixel, end_pixels, bbox_mask, endpixel_avail_dirs)

        #if not endpixel_avail_dirs:
        if endpixel_dir is None:
            return image, bbox_mask

        # Select end pixel    
        end_pixel = end_pixels[endpixel_dir]

        close_dirs = np.arange(endpixel_dir-45, endpixel_dir+45)%360
        
        #close_dirs = close_dirs.tolist() + np.arange(endpixel_dir+180-45,endpixel_dir+180+45).tolist()
        
        close_dirs = close_dirs.tolist()
        endpixel_avail_dirs = [ele for ele in endpixel_avail_dirs if ele not in close_dirs]

        if self.rng.binomial(n=1, p=self.corner_prob) and endpixel_avail_dirs:
            
            corner_endpixel_dir, endpixel_avail_dirs = self.line_end(start_pixel, end_pixels, bbox_mask, endpixel_avail_dirs)
        
            if corner_endpixel_dir is None:
                return image, bbox_mask
            
            corner_endpixel = end_pixels[corner_endpixel_dir]

            edge_ratio = self.rng.uniform(0,1, size=1)[0]

            corner_endpixel = np.squeeze(np.round(np.dot(np.vstack((start_pixel, corner_endpixel)).T, 
                np.array([1-edge_ratio, edge_ratio]).reshape(2,1))).astype(np.int32))

            image = cv2.line(image, (start_pixel[0], start_pixel[1]), (corner_endpixel[0], corner_endpixel[1]), color = (255), thickness=1)
            bbox_mask = cv2.line(bbox_mask, (start_pixel[0], start_pixel[1]), (corner_endpixel[0], corner_endpixel[1]), color = (255), thickness=2)
        
        image = cv2.line(image, (start_pixel[0], start_pixel[1]), (end_pixel[0], end_pixel[1]), color = (255), thickness=1)
        bbox_mask = cv2.line(bbox_mask, (start_pixel[0], start_pixel[1]), (end_pixel[0], end_pixel[1]), color = (255), thickness=2)

        return image, bbox_mask

class ClusterNoiseSimulator():

    def __init__(self, rng, rad_range):
        self.rng = rng
        self.rad_range = rad_range

    def generate_noise_cluster(self, image, bbox_mask):
        
        # Radial noise are center candidates
        available_pixels = np.concatenate((np.where(bbox_mask == 0)[1].reshape(-1,1), np.where(bbox_mask == 0)[0].reshape(-1,1)), axis=-1)
        # Randomly selected center
        center = random.choice(available_pixels)
        # Randomly generated axes and angle
        radius = self.rng.integers(self.rad_range[0], self.rad_range[1], size=1, endpoint = True)[0]
        axes = [radius, np.round(radius*self.rng.uniform(0.85,1,size=1)[0]).astype(np.int32)]
        angle = self.rng.integers(0, 180, size=1, endpoint = True)[0]

        # Generate oval noise area
        oval_noise = np.zeros(shape = bbox_mask.shape)
        oval_noise = cv2.ellipse(oval_noise, center, axes, angle,
                                 startAngle=0, endAngle=360, color=255, thickness=-1)

        # Check for intersection with other noisy objects and target oval objects
        overlay = np.sum(np.multiply(bbox_mask,oval_noise))

        # Until noisy cluster is intersecting other objects reduce it's size
        while (overlay > 0) and (min(axes)>1):
            
            axes[0] //= 2
            axes[1] //= 2
            oval_noise = np.zeros(shape = bbox_mask.shape)
            oval_noise = cv2.ellipse(oval_noise, center, axes, angle,
                                    startAngle=0, endAngle=360, color=255, thickness=-1)

            overlay = np.sum(np.multiply(bbox_mask, oval_noise))

        if min(axes) > 5:

            polygon = cv2.ellipse2Poly(center, axes, angle, 0, 360, 1)
            bbox = np.min(polygon, axis = 0).tolist() + np.max(polygon, axis = 0).tolist()
            bbox_mask = cv2.rectangle(bbox_mask,(bbox[0], bbox[1]), (bbox[2], bbox[3]),(255),-1)

            noise = self.rng.uniform(0,1, size=bbox_mask.shape) > 0.99
            oval_noise = np.multiply(noise, oval_noise)
            image += oval_noise
        
        return image, bbox_mask