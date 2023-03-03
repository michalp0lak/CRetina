# %%
# importing cv2
import cv2
import numpy as np  
import numba
import random

def calculate_ellipse_pixels(center, axes, angle):
    
    Cos = np.cos(np.deg2rad(angle))
    Sin = np.sin(np.deg2rad(angle))

    coords = []

    for ang in range(0,360):

        u = np.cos(np.deg2rad(ang))*axesLength[0]
        v = np.sin(np.deg2rad(ang))*axesLength[1]

        x = u*Cos - v*Sin + center_coordinates[0]
        y = u*Sin - v*Cos + center_coordinates[1]
        triplets.append([x,y,ang])

    return np.round(np.array(coords),0).astype(np.int32)

def ellipse_extremas(center_coordinates,axesLength,angle):

    extremas = np.zeros(shape=(4,))

    Sin = np.sin(np.deg2rad(angle))
    Cos = np.cos(np.deg2rad(angle))

    tx = np.arctan(-(axesLength[1]*Sin)/(axesLength[0]*Cos))
    ty = np.arctan(-(axesLength[1]*Cos)/(axesLength[0]*Sin))

    x1 = axesLength[0]*Cos*np.cos(tx) - axesLength[1]*Sin*np.sin(ty) + center_coordinates[0]
    y1 = axesLength[0]*Sin*np.cos(tx) - axesLength[1]*Cos*np.sin(ty) + center_coordinates[1]
    x2 = axesLength[0]*Cos*np.cos(tx + np.pi) - axesLength[1]*Sin*np.sin(ty + np.pi) + center_coordinates[0]
    y2 = axesLength[0]*Sin*np.cos(tx + np.pi) - axesLength[1]*Cos*np.sin(ty + np.pi) + center_coordinates[1]

    if x1 < x2:
        extremas[0] = np.round(x1,0).astype(np.int32)
        extremas[2] = np.round(x2,0).astype(np.int32)
    else:
        extremas[0] = np.round(x2,0).astype(np.int32)
        extremas[2] = np.round(x1,0).astype(np.int32)

    if y1 < y2:
        extremas[1] = np.round(y1,0).astype(np.int32)
        extremas[3] = np.round(y2,0).astype(np.int32)
    else:
        extremas[1] = np.round(y2,0).astype(np.int32)
        extremas[3] = np.round(y1,0).astype(np.int32)


    return [int(ext) for ext in extremas]

def circle_center_limits(radius, width, height):

    assert width//2 > radius, 'Circle does not fit in image'
    assert height//2 > radius, 'Circle does not fit in image'
    
    return (radius, width-radius), (radius, height-radius)
    
def ellipse_center_limits(axesLength, angle, width, height):

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
        x_up = image.shape[1]- np.round(np.max([x1,x2]))

        y_low = np.round(np.abs(np.min([y1,y2])))
        y_up = image.shape[0]- np.round(np.max([y1,y2]))

        return (int(x_low), int(x_up)), (int(y_low), int(y_up))


def generate_oval_shape(width, height, radius_limit, ellipse_aspect_ratio, circle_prob = 0.2):

    # Circle = 1 X Ellipse = 0
    oval_shape = rng.binomial(n=1, p = circle_prob)

    # Radius
    radius = rng.integers(low=radius_limit[0], high=radius_limit[1], size=1, endpoint = True)[0]

    # Generate circle attributes
    if oval_shape:

        x_center_limit, y_center_limit = circle_center_limits(radius, width, height)
        cx = rng.integers(low=x_center_limit[0], high=x_center_limit[1], size=1, endpoint = True)[0]
        cy = rng.integers(low=y_center_limit[0], high=y_center_limit[1], size=1, endpoint = True)[0]
        center_coordinates = (cx, cy) 
        axesLength = (radius, radius)
        angle = 0

    # Generate ellipse attributes
    else:

        aspect_ratio = rng.uniform(ellipse_aspect_ratio[0], ellipse_aspect_ratio[1], 1)
        axesLength = (radius,np.ceil(radius*aspect_ratio).astype(np.int32)[0])
        
        angle = rng.uniform(0, np.pi, 1)[0]

        x_center_limit, y_center_limit = ellipse_center_limits(axesLength, angle, width, height)
        cx = rng.integers(low=x_center_limit[0], high=x_center_limit[1], size=1, endpoint = True)[0]
        cy = rng.integers(low=y_center_limit[0], high=y_center_limit[1], size=1, endpoint = True)[0]
        center_coordinates = (cx, cy) 

    # Get bounding bix
    polygon = cv2.ellipse2Poly(center_coordinates, axesLength, int(np.rad2deg(angle)), 0, 360, 1)
    bbox = np.min(polygon, axis = 0).tolist() + np.max(polygon, axis = 0).tolist()

    return {'class': oval_shape,'center': center_coordinates, 'axes':axesLength, 'angle': angle, 'bbox': bbox}


def visualize(image,oval_objects):

    img_canvas = image.copy()

    for oval in oval_objects:

        # Draw a ellipse with blue line borders of thickness of -1 px
        img_canvas = cv2.ellipse(img_canvas, oval['center'], oval['axes'], np.rad2deg(oval['angle']),
                                startAngle = 0, endAngle = 360, color = (255,255,255), thickness = 1)
        # Draw bounding box
        img_canvas = cv2.rectangle(img_canvas,(oval['bbox'][0],oval['bbox'][1]),(oval['bbox'][2],oval['bbox'][3]),(255,0,0),1)

        # Draw center
        img_canvas = cv2.circle(img_canvas, oval['center'], 0, color=(0,0,255), thickness=6)

        # Draw axes
        if oval['axes'][0] == oval['axes'][1]:
            img_canvas = cv2.line(img_canvas, (oval['center'][0], oval['center'][1]), 
                                (oval['center'][0]+oval['axes'][0],oval['center'][1]), 
                                color = (0,255,255), thickness=1) 
        else:
            rot_mat = np.array([[np.cos(oval['angle']), -np.sin(oval['angle'])],
                                [np.sin(oval['angle']), np.cos(oval['angle'])]])
        
            end_points = np.round(np.array([[oval['axes'][0],0],
                                [0,oval['axes'][1]]])).astype(np.int32)
        
            end_points = np.dot(rot_mat,end_points).T + oval['center']
         
            img_canvas = cv2.line(img_canvas, (oval['center'][0], oval['center'][1]), 
                                (int(end_points[0,0]), int(end_points[0,1])),
                                color = (0,255,255), thickness=1)
        
            img_canvas = cv2.line(img_canvas, (oval['center'][0], oval['center'][1]), 
                                (int(end_points[1,0]), int(end_points[1,1])), 
                                color = (255,255,0), thickness=1)
    

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("image", 1300, 1600)
    cv2.imshow('image', img_canvas)
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 

def gen_img_dim(rng, pixel_limit = (500, 2000), img_ratio_limit = (0.5,1)):

    # Generate image bigger dimension
    img_size = rng.integers(low=pixel_limit[0], high=pixel_limit[1], size=1, endpoint = True)[0]
    # Generate ratio to get smaller dimension
    aspect_ratio = rng.uniform(img_ratio_limit[0], img_ratio_limit[1], 1)
    # Generate image orientation -> determine if bigger dim is width or height
    orientation = rng.binomial(n=1, p = 0.5)

    if orientation == 1:
        width = img_size
        height = np.ceil(img_size*aspect_ratio).astype(np.int32)[0]
    elif orientation == 0:
        height = img_size
        width = np.ceil(img_size*aspect_ratio).astype(np.int32)[0]

    return (height, width, 3)

@numba.jit(nopython=True)
def iou_jit(boxes, query_boxes, eps=0.0):
    """calculate box iou. note that jit version runs 2x faster than cython in 
    my machine!
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float32)
    for k in range(K):
        box_area = ((query_boxes[k, 2] - query_boxes[k, 0] + eps) *
                    (query_boxes[k, 3] - query_boxes[k, 1] + eps))
        for n in range(N):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) -
                  max(boxes[n, 0], query_boxes[k, 0]) + eps)
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) -
                      max(boxes[n, 1], query_boxes[k, 1]) + eps)
                if ih > 0:
                    ua = (
                        (boxes[n, 2] - boxes[n, 0] + eps) *
                        (boxes[n, 3] - boxes[n, 1] + eps) + box_area - iw * ih)
                    overlaps[n, k] = iw * ih / ua

    return overlaps

def occlusion_range(occlusion_pct_range=(0.25, 0.5)):

    # Randomly generate occlusion proportion with triangular distribution
    #occlusion_proportion_tri = rng.triangular(occlusion_pct_range[0],
    #                                        occlusion_pct_range[1],
    #                                        occlusion_pct_range[1])

    # Randomly generate occlusion proportion with power distribution
    #occlusion = rng.power(3)*(occlusion_pct_range[1]-occlusion_pct_range[0]) + occlusion_pct_range[0]

    # Randomly generate occlusion proportion with beta distribution
    occlusion = rng.beta(2,1.5)*(occlusion_pct_range[1]-occlusion_pct_range[0]) + occlusion_pct_range[0]
    # Get start and end point of occlusion
    occlusion_start_point = int(np.rad2deg(rng.integers(low=0, high=2*np.pi, size=1, endpoint = False)[0]))
    occlusion_end_point = int(np.rad2deg(2*np.pi*(1-occlusion))) + occlusion_start_point

    return occlusion_start_point, occlusion_end_point

def occluded_orientation(occlusion_start_point, occlusion_end_point):

    # Get not occluded part of ellipse/circle
    deg_ran = np.arange(occlusion_start_point, occlusion_end_point)
    # Normalize in <0,360>
    deg_ran[deg_ran >= 360] = deg_ran[deg_ran >= 360] -360
    # Sector frequencies
    rad_cover = [sum((0 <= deg_ran) & (deg_ran < 90)),
                sum((90 <= deg_ran) & (deg_ran < 180)),
                sum((180 <= deg_ran) & (deg_ran < 270)),
                sum((270 <= deg_ran) & (deg_ran < 360))
                ]

    # Determine orientation based on sector occupancy
    order = np.argsort(rad_cover)[::-1]+1
    orientation_index = order[0]*order[1]

    if orientation_index == 12: orientation = 1
    elif orientation_index == 2: orientation = 2
    elif orientation_index == 6: orientation = 3
    elif orientation_index == 4: orientation = 4

    return orientation

def generate_sample():

    oval_obj_limit = (30,100)
    oval_obj_num = rng.integers(low=oval_obj_limit[0], high=oval_obj_limit[1], size=1, endpoint = True)[0]

    sample = []
    current_boxes = []

    for i in range(oval_obj_num):

        if len(current_boxes) > 0:

            oval = generate_oval_shape(width, height, radius_limit, 
                            ellipse_aspect_ratio = (0.5,1), circle_prob = 0.5)

            iou = iou_jit(np.array(current_boxes).reshape(-1,4), np.array(oval['bbox']).reshape(-1,4))
  
            if np.all(iou == 0):
                current_boxes.append(oval['bbox'])
                sample.append(oval)

        else:
            oval = generate_oval_shape(width, height, radius_limit, 
                                        ellipse_aspect_ratio = (0.5,1), circle_prob = 0.5)
            current_boxes.append(oval['bbox'])
            sample.append(oval)

    return sample


# Parameters simulation
rng = np.random.default_rng()

# Image generation input parameters
pixel_limit = (1000, 4000)
img_ratio_limit = (0.5,1)
###################################

image = np.zeros(gen_img_dim(rng), np.uint8)
height, width = image.shape[:2]
img_canvas = image.copy()

# Geometrical object size parameter
radius_limit = (10,100)
###################################
assert np.ceil(pixel_limit[0]*img_ratio_limit[0])//2 > radius_limit[1], ('Wrong configuration -> cases when circle/ellipse wont fit in image '
'can occur. Check simulation setting: You can increase minimal image size or decrease maximal radius size')

#ksize = (5, 5)
#  
## Using cv2.blur() method 
#image = cv2.blur(image, ksize, cv2.BORDER_DEFAULT) 

sample = generate_sample()
visualize(image, sample)

# Parameters simulation
# %%
import cv2
import numpy as np  
import numba
import random
import json
import yaml
import time

@numba.jit(nopython=True)
def iou_jit(boxes, query_boxes, eps=0.0):
    """calculate box iou. note that jit version runs 2x faster than cython in 
    my machine!
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float32)
    for k in range(K):
        box_area = ((query_boxes[k, 2] - query_boxes[k, 0] + eps) *
                    (query_boxes[k, 3] - query_boxes[k, 1] + eps))
        for n in range(N):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + eps)
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + eps)
                if ih > 0:
                    ua = (
                        (boxes[n, 2] - boxes[n, 0] + eps) *
                        (boxes[n, 3] - boxes[n, 1] + eps) + box_area - iw * ih)
                    overlaps[n, k] = iw * ih / ua

    return overlaps

def bresenham(x0, y0, x1, y1):
    """Yield integer coordinates on the line from (x0, y0) to (x1, y1).
    Input coordinates should be integers.
    The result will contain both the start and the end point.
    """
    dx = x1 - x0
    dy = y1 - y0

    xsign = 1 if dx > 0 else -1
    ysign = 1 if dy > 0 else -1

    dx = abs(dx)
    dy = abs(dy)

    if dx >= dy:
        xx, xy, yx, yy = xsign, 0, 0, ysign
    else:
        dx, dy = dy, dx
        xx, xy, yx, yy = 0, ysign, xsign, 0

    D = 2*dy - dx
    y = 0

    for x in range(dx + 1):
        yield x0 + x*xx + y*yx, y0 + x*xy + y*yy
        if D >= 0:
            y += 1
            D -= 2*dx
        D += 2*dy

#####################################################################################################################
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

            direction = (opening_angle+closing_angle)//2
            if direction>360: direction -= 360

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

        # Get bounding bix
        polygon = cv2.ellipse2Poly(center_coordinates, axesLength, int(np.rad2deg(angle)), 0, 360, 1)
        bbox = (np.min(polygon, axis = 0)-line_t).tolist() + (np.max(polygon, axis = 0)+line_t).tolist()

        oval_start, oval_end, orientation = self.occlusion()

        orientation += angle
        if orientation > 360: orientation -= 360

        return {'class': oval_shape,'center': center_coordinates, 'axes':axesLength, 'angle': angle, 'bbox': bbox,
                'occlusion': [oval_start, oval_end], 'orientation': orientation,'line': line_t}

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

        close_dirs = np.arange(endpixel_dir-45, endpixel_dir+45)
        close_dirs[close_dirs < 0] += 360
        close_dirs[close_dirs > 360] -= 360
        
        #close_dirs = close_dirs.tolist() + np.arange(endpixel_dir+180-45,endpixel_dir+180+45).tolist()
        
        close_dirs = close_dirs.tolist()
        endpixel_avail_dirs = [ele for ele in endpixel_avail_dirs if ele not in close_dirs]

        if self.rng.binomial(n=1, p=self.corner_prob) and endpixel_avail_dirs:
            
            st = time.time()
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


class ImageAugmentor():

    def __init__(self, rng, augmentatation_cfg):

        self.rng = rng
        self.border_prob = augmentatation_cfg['border_distortion']['prob']
        self.border_intensity = augmentatation_cfg['border_distortion']['intensity_range']
        self.blur_prob = augmentatation_cfg['blur']['prob']
        self.blur_kernel_size = augmentatation_cfg['blur']['kernel_size']
        self.noise_cfg = augmentatation_cfg['noise']
        
        self.LNS = LinearNoiseSimulator(self.rng, **self.noise_cfg['linear'])
        self.CNS = ClusterNoiseSimulator(self.rng, **self.noise_cfg['cluster'])
        self.noisy_obj_prob = self.noise_cfg['type_prob']


    def augment(self, image, ovals, objs_num):

        bbox_mask = np.zeros(shape = image.shape)

        for oval in ovals:
            bbox_mask = cv2.rectangle(bbox_mask,(oval['bbox'][0],oval['bbox'][1]),(oval['bbox'][2],oval['bbox'][3]),(255),-1)
 
        # Add noisy geometrical objects
        for i in range(objs_num):
            
            noisy_obj_index = np.argmax(self.rng.multinomial(1, self.noisy_obj_prob))

            if noisy_obj_index == 0:
                image, bbox_mask = self.LNS.generate_linear_noise(image, bbox_mask)
            elif noisy_obj_index == 1:
                image, bbox_mask = self.CNS.generate_noise_cluster(image, bbox_mask)

        # Add random noise
        free_pixel_mask = bbox_mask == 0
        random_pixel_noise = self.rng.uniform(0,1,size=image.shape) > 0.999
        image += np.multiply(free_pixel_mask, random_pixel_noise)

        if self.rng.binomial(n=1, p = self.border_prob):
            
            image = self.rng.uniform(0,1, size=image.shape) * (image>0)

        if self.rng.binomial(n=1, p = self.blur_prob):

            ksize = self.rng.integers(self.blur_kernel_size[0], self.blur_kernel_size[1], size=1, endpoint = True)[0]
            image = cv2.blur(image, (ksize,ksize), cv2.BORDER_DEFAULT) 

        return image



class Simulation():

    def __init__(self, simulation_config):

        self.rng = np.random.default_rng()
        self.simulation_config = simulation_config
        self.output_path = simulation_config['general']['output_path']
        self.augmentation_cfg = simulation_config['augmentation']

        self.sample_size = simulation_config['general']['sample_size']
        self.sample_split = simulation_config['general']['sample_split']
        self.train_set, self.val_set, self.test_set = self.data_split()

        self.pixel_limit = simulation_config['general']['pixel_limit']
        self.img_aspect_ratio = simulation_config['general']['img_aspect_ratio']
        self.OSIM = OvalSimulator(self.rng, simulation_config['oval_simulator'])
        self.Augmentor = ImageAugmentor(self.rng, simulation_config['augmentation'])

    def data_split(self):

        sample_indexes = np.arange(0,self.sample_size)
        np.random.shuffle(sample_indexes)

        train_thresh = int(np.round(self.sample_size*self.sample_split[0]))
        val_thresh = train_thresh + int(np.round(self.sample_size*self.sample_split[1]))

        training_indexes = sample_indexes[:train_thresh].tolist()
        validation_indexes = sample_indexes[train_thresh:val_thresh].tolist()
        testing_indexes = sample_indexes[val_thresh:].tolist()

        return training_indexes, validation_indexes, testing_indexes

    def simulate_image_dimensions(self):

        # Generate image bigger dimension
        img_size = self.rng.integers(low=self.pixel_limit[0], high=self.pixel_limit[1], size=1, endpoint = True)[0]
        # Generate ratio to get smaller dimension
        aspect_ratio = self.rng.uniform(self.img_aspect_ratio[0], self.img_aspect_ratio[1], 1)
        # Generate image orientation -> determine if bigger dim is width or height
        orientation = self.rng.binomial(n=1, p = 0.5)

        if orientation == 1:
            width = img_size
            height = np.ceil(img_size*aspect_ratio).astype(np.int32)[0]
        elif orientation == 0:
            height = img_size
            width = np.ceil(img_size*aspect_ratio).astype(np.int32)[0]

        return (height, width)

    def get_sample(self):

        h,w = self.simulate_image_dimensions()

        # Here generate number range for ovals and noisy objs based on image size
        E = (h*w)**(0.28)
        var = E**(0.5)-1

        oval_num = int(np.round(self.rng.normal(E, var, 1)))
        noise_objs_num = int(np.round(self.rng.normal(E/2, var, 1)))
        print(oval_num, noise_objs_num)

        image = np.zeros(shape=(h,w))
        ovals = self.OSIM.generate_ovals(image, oval_num)

        for oval in ovals:

            image = cv2.ellipse(image, oval['center'], oval['axes'], np.rad2deg(oval['angle']),
                                    startAngle = oval['occlusion'][0], endAngle = oval['occlusion'][1], 
                                    color = (255), thickness = oval['line'])

        image = self.Augmentor.augment(image, ovals, noise_objs_num)

        return image, ovals

    def visualize_sample(self, image, oval_objects):

        img_canvas = cv2.merge((image,image,image))

        for oval in oval_objects:

            # Draw a ellipse with blue line borders of thickness of -1 px
            img_canvas = cv2.ellipse(img_canvas, oval['center'], oval['axes'], np.rad2deg(oval['angle']),
                                    startAngle = oval['occlusion'][0], endAngle = oval['occlusion'][1], 
                                    color = (255,255,255), thickness = oval['line'])
            # Draw bounding box
            img_canvas = cv2.rectangle(img_canvas,(oval['bbox'][0],oval['bbox'][1]),(oval['bbox'][2],oval['bbox'][3]),(255,0,0),1)

            # Draw center
            img_canvas = cv2.circle(img_canvas, oval['center'], 0, color=(0,0,255), thickness=6)

            # Draw axes
            if oval['axes'][0] == oval['axes'][1]:
                img_canvas = cv2.line(img_canvas, (oval['center'][0], oval['center'][1]), 
                                    (oval['center'][0]+oval['axes'][0],oval['center'][1]), 
                                    color = (0,255,255), thickness=1) 
            else:
                rot_mat = np.array([[np.cos(oval['angle']), -np.sin(oval['angle'])],
                                    [np.sin(oval['angle']), np.cos(oval['angle'])]])
            
                end_points = np.round(np.array([[oval['axes'][0],0],
                                    [0,oval['axes'][1]]])).astype(np.int32)
            
                end_points = np.dot(rot_mat,end_points).T + oval['center']
            
                img_canvas = cv2.line(img_canvas, (oval['center'][0], oval['center'][1]), 
                                    (int(end_points[0,0]), int(end_points[0,1])),
                                    color = (0,255,255), thickness=1)
            
                img_canvas = cv2.line(img_canvas, (oval['center'][0], oval['center'][1]), 
                                    (int(end_points[1,0]), int(end_points[1,1])), 
                                    color = (255,255,0), thickness=1)
        

        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("image", 2000, 2000)
        cv2.imshow('image', img_canvas)
        cv2.waitKey(0) 
        cv2.destroyAllWindows()

    def simulate_dataset(self):

        if not os.path.exists(self.output_path + '/training/'): 
            os.makedirs(self.output_path + '/training/')
        if not os.path.exists(self.output_path + '/validation/'): 
            os.makedirs(self.output_path + '/validation/')
        if not os.path.exists(self.output_path + '/testing/'): 
            os.makedirs(self.output_path + '/testing/')

        for i in range(self.sample_size):
            
            image, ovals = self.get_sample()

            if i in self.train_set:

                cv2.imwrite(self.output_path + '/training/{}.png'.format(i), np.squeeze(image[:,:,0]))
                with open(self.output_path + "/training/annot_{}.json".format(i), "w") as fp:
                    json.dump(oval,fp) 

            if i in self.test_set:
                cv2.imwrite(self.output_path + '/testing/{}.png'.format(i), np.squeeze(image[:,:,0]))
                with open(self.output_path + "/testing/annot_{}.json".format(i), "w") as fp:
                    json.dump(oval,fp) 

            if i in self.val_set:
                cv2.imwrite(self.output_path + '/validation/{}.png'.format(i), np.squeeze(image[:,:,0]))
                with open(self.output_path + "/validation/annot_{}.json".format(i), "w") as fp:
                    json.dump(oval,fp) 


        with open(self.output_path + '/simulation_config.json', "w") as outfile:
            json.dump(dict(self.simulation_config), outfile)



with open('./config.yml') as f: cfg = yaml.safe_load(f)

sim = Simulation(cfg)

#sim.simulate_dataset()
img, objs = sim.get_sample()
sim.visualize_sample(img, objs)


# %%
#########################################################
# Bresenham - Opencv draw line equality test
#########################################################
from bresenham import bresenhamline

rng = np.random.default_rng()
A = rng.integers(0,1000, size=(10000,2))
B = rng.integers(0,1000, size=(10000,2))

for i in range(A.shape[0]):

    mask1 = np.zeros(shape=(1000,1000))
    mask2 = np.zeros(shape=(1000,1000))

    mask1 = cv2.line(mask1, (A[i][0],A[i][1]),(B[i][0],B[i][1]), (255), 1)
    pixels = bresenhamline(A[i].reshape(-1,2), B[i].reshape(-1,2), max_iter=-1)
    for pix in pixels: mask2[pix[1],pix[0]] = 255
    mask2[A[i][1],A[i][0]] = 255
    #mask2[B[i][1],B[i][0]] = 255
    print(np.sum(mask1!=mask2))
    #if np.sum(mask1!=mask2) != 0:
    #    print(A[i],B[i])
# %%
#########################################################
# Bresenham - Opencv draw line equality visualization
#########################################################
from bresenham import bresenhamline
import numpy as np
import cv2

rng = np.random.default_rng()
A = rng.integers(0,1000, size=(1,2))[0]
B = rng.integers(0,1000, size=(1,2))[0]

mask1 = np.zeros(shape=(1000,1000))
mask2 = np.zeros(shape=(1000,1000))

mask1 = cv2.line(mask1, (A[0],A[1]),(B[0],B[1]), 255, 1)

pixels = bresenhamline(A.reshape(-1,2), B.reshape(-1,2), max_iter=-1)
for pix in pixels: mask2[pix[1],pix[0]] = 255

mask2[A[1],A[0]] = 255
#mask2[B[1],B[0]] = 255

diff = (mask1!=mask2).astype(np.float64)*255
print(np.sum(mask1!=mask2))
cv2.namedWindow("diff", cv2.WINDOW_NORMAL)
cv2.resizeWindow("diff", 2000, 2000)
cv2.imshow('diff', diff)
cv2.waitKey(0) 
cv2.destroyAllWindows()

# %%
endpixel_dir = 12
close_dirs = np.arange(endpixel_dir-45, endpixel_dir+45)
close_dirs
# %%
close_dirs%360