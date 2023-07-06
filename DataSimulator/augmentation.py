import cv2
import numpy as np
from shapes import LinearNoiseSimulator, ClusterNoiseSimulator

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

    def augment(self, image, objs, objs_num):

        bbox_mask = np.zeros(shape = image.shape)

        for oval in objs:
            bbox_mask = cv2.rectangle(bbox_mask,(oval['bbox'][0],oval['bbox'][1]),(oval['bbox'][2],oval['bbox'][3]),(255),-1)
 
        # Add noisy geometrical objects
        for i in range(objs_num):

            noisy_obj_index = np.argmax(self.rng.multinomial(1, self.noisy_obj_prob))

            if noisy_obj_index == 0:
                image, bbox_mask = self.LNS.generate_linear_noise(image, bbox_mask)
            elif noisy_obj_index == 1:
                image, bbox_mask = self.CNS.generate_noise_cluster(image, bbox_mask)
                
        # Add random noise
        #free_pixel_mask = bbox_mask == 0
        #random_pixel_noise = self.rng.uniform(0,1,size=image.shape) > 0.999
        #image += np.multiply(free_pixel_mask, random_pixel_noise)*255

        if self.rng.binomial(n=1, p = self.border_prob):
            
            image = self.rng.uniform(self.border_intensity,1, size=image.shape) * (image>0)

        if self.rng.binomial(n=1, p = self.blur_prob):

            ksize = self.rng.integers(self.blur_kernel_size[0], self.blur_kernel_size[1], size=1, endpoint = True)[0]
            image = cv2.blur(image, (ksize,ksize), cv2.BORDER_DEFAULT) 

        return image