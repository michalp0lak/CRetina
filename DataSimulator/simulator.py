import os
import cv2
import numpy as np  
import random
import json
import yaml
from datetime import datetime
from ops import calculate_ellipse_pixel
from augmentation import ImageAugmentor
from shapes import OvalSimulator

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

        E = (h*w)**(0.28)
        var = E**(0.5)-1

        oval_num = int(np.round(self.rng.normal(E, var, 1)))
        noise_objs_num = int(np.round(self.rng.normal(E/2, var, 1)))

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
            img_canvas = cv2.rectangle(img_canvas,(oval['label_bbox'][0],oval['label_bbox'][1]),
                                      (oval['label_bbox'][2],oval['label_bbox'][3]),(255,0,0),1)

            # Draw center
            img_canvas = cv2.circle(img_canvas, oval['center'], 0, color=(0,0,255), thickness=6)

            # Draw axes
            if oval['axes'][0] == oval['axes'][1]:
                
                endpoint = calculate_ellipse_pixel(oval['center'], oval['axes'], 0, oval['orientation']-90)[0]
                
                img_canvas = cv2.line(img_canvas, (oval['center'][0], oval['center'][1]), 
                                    (endpoint[0],endpoint[1]), 
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

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.output_path = self.output_path + '/' + timestamp

        if not os.path.exists(self.output_path + '/training/'): 
            os.makedirs(self.output_path + '/training/')
        if not os.path.exists(self.output_path + '/validation/'): 
            os.makedirs(self.output_path + '/validation/')
        if not os.path.exists(self.output_path + '/testing/'): 
            os.makedirs(self.output_path + '/testing/')

        for i in range(self.sample_size):
            
            image, ovals = self.get_sample()

            if i in self.train_set:

                cv2.imwrite(self.output_path + '/training/{}.png'.format(i), image)
                with open(self.output_path + "/training/annot_{}.json".format(i), "w") as fp:
                    json.dump(ovals,fp) 

            if i in self.test_set:
                cv2.imwrite(self.output_path + '/testing/{}.png'.format(i), image)
                with open(self.output_path + "/testing/annot_{}.json".format(i), "w") as fp:
                    json.dump(ovals,fp) 

            if i in self.val_set:
                cv2.imwrite(self.output_path + '/validation/{}.png'.format(i), image)
                with open(self.output_path + "/validation/annot_{}.json".format(i), "w") as fp:
                    json.dump(ovals,fp) 


        with open(self.output_path + '/simulation_config.json', "w") as outfile:
            json.dump(dict(self.simulation_config), outfile)


if __name__== '__main__':

    with open('./config.yml') as f: cfg = yaml.safe_load(f)
    simulator = Simulation(cfg)
    #image, oval_objects = simulator.get_sample()
    #simulator.visualize_sample(image, oval_objects)
    simulator.simulate_dataset()