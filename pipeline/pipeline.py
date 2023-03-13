import logging
import re
from datetime import datetime
import os 
from os.path import join, exists, dirname, abspath
import random
import math
import yaml
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import cv2
import time

from torch.utils.data import DataLoader
from dataset.dataloaders import TorchDataloader, ConcatBatcher
from pipeline.utils import latest_ckpt
from utils import make_dir
from pipeline.metrics import MetricEvaluator
from pipeline.base_pipeline import BasePipeline

log = logging.getLogger(__name__)

class ObjectDetection(BasePipeline):
    """Pipeline for object detection."""

    def __init__(self, model, dataset, global_cfg, **kwargs):

        super().__init__(model=model,
                         dataset=dataset,
                         global_cfg=global_cfg,
                         **kwargs)
                         
        self.ME = MetricEvaluator(self.device)

    def save_ckpt(self, epoch, save_best = False):

        ckpt_dir = join(self.cfg.log_dir, 'checkpoint/')
        make_dir(ckpt_dir)

        if save_best: path = join(ckpt_dir,'ckpt_best.pth')
        else: path = join(ckpt_dir, f'ckpt_{epoch:05d}.pth')

        torch.save(
            dict(epoch=epoch,
                 model_state_dict=self.model.state_dict(),
                 optimizer_state_dict=self.optimizer.state_dict()),
                #scheduler_state_dict=self.scheduler.state_dict()),
                 path)

        log.info(f'Epoch {epoch:3d}: save ckpt to {path:s}')

    def load_ckpt(self):
        
        ckpt_dir = join(self.cfg.log_dir, 'checkpoint/')
        epoch = 0
        
        if not self.cfg.inference_mode:
            
            if self.cfg.is_resume:

                last_ckpt_path = latest_ckpt(ckpt_dir)
            
                if last_ckpt_path:

                    epoch = int(re.findall(r'\d+', last_ckpt_path)[-1])
                    ckpt_path = last_ckpt_path
                    log.info('Model restored from the latest checkpoint: {}'.format(epoch))

                else:

                    log.info('Latest checkpoint was not found')
                    log.info('Initializing from scratch.')
                    return epoch, None

            else:
                log.info('Initializing from scratch.')
                return epoch, None    

        else:

            ckpt_path = self.cfg.log_dir + 'checkpoint/ckpt_best.pth'

            if not os.path.exists(ckpt_path):
                raise ValueError('There is not pretrained model for inference. Best output of training should be found as {}'.format(ckpt_path))
            
            print(f'Loading checkpoint {ckpt_path}')
        
        log.info(f'Loading checkpoint {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location=self.device) 

        self.model.load_state_dict(ckpt['model_state_dict'])

        if 'optimizer_state_dict' in ckpt and hasattr(self, 'optimizer'):
            log.info('Loading checkpoint optimizer_state_dict')
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        if 'scheduler_state_dict' in ckpt and hasattr(self, 'scheduler'):
            log.info('Loading checkpoint scheduler_state_dict')
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])

        return epoch, ckpt_path

    def transform_input_batch(self, boxes, centers, radius, dirs, labels=None):

        dicts = []
       
        if labels is None:
            for box, center, rad, direction in zip(boxes, centers, radius, dirs):
                dicts.append({'box': box, 'center': center, 'radius': rad, 'direction': direction})
        else:
            for box, center, rad, direction, label in zip(boxes, centers, radius, dirs, labels):
                dicts.append({'label': label, 'box': box, 'center': center, 'radius': rad, 'direction': direction})

        return dicts

    def transform_for_metric(self, boxes):

        """Convert data for evaluation:
        Args:
            bboxes: List of predicted items (box, label).
        """

        box_dicts = {
            'label': torch.empty((len(boxes),)).to(self.device),
            'score': torch.empty((len(boxes),)).to(self.device),
            'box': torch.empty((len(boxes), 4)).to(self.device),
            'center': torch.empty((len(boxes), 2)).to(self.device),
            'radius': torch.empty((len(boxes),)).to(self.device),
            'direction': torch.empty((len(boxes),)).to(self.device)
                    }

        for i in range(len(boxes)):
            box_dict = boxes[i]

            for k in box_dict:
                box_dicts[k][i] = box_dict[k]

        return box_dicts

    def adjust_learning_rate(self, optimizer, gamma, epoch, step_index, iteration, epoch_size):

        """Sets the learning rate
        # Adapted from PyTorch Imagenet example:
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py
        """
        warmup_epoch = -1
        if epoch <= warmup_epoch:
            lr = 1e-6 + (self.cfg.optimizer.lr-1e-6) * iteration / (epoch_size * warmup_epoch)
        else:
            lr = self.cfg.optimizer.lr * (self.cfg.optimizer.gamma ** (step_index))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr


    def run_inference(self, data):
        """Run inference on given data.
        Args:
            data: A raw data.
        Returns:
            Returns the inference results.
        """

        self.load_ckpt()
        self.model.eval()

        # If run_inference is called on raw data.
        if isinstance(data, dict):
            batcher = ConcatBatcher(self.device)
            data = batcher.collate_fn([{
                'data': data['data'],
                'attr': data['attr']
            }])

        data.to(self.device)
        self.model.head.image_size = data.images.shape[-2:]

        with torch.no_grad():

            results = self.model(data)
            boxes = self.model.inference_end(results)

        return boxes

    def show_inference(self):

        test_dataset = self.dataset.get_split('testing')
        
        test_split = TorchDataloader(dataset=test_dataset,
                                preprocess=self.model.preprocess,
                                transform=self.model.transform,
                                use_cache=False,
                                shuffle=False)

        idx = random.sample(range(0, len(test_dataset)), 1)
        idx = [50]
        data_item = test_split.__getitem__(idx[0])
        print(idx[0])
        print(data_item['attr'])

        predicted_items = self.run_inference(data_item)
    
        data = data_item['data']
        print(data['image'].shape)          
        target = [self.transform_for_metric(self.transform_input_batch(boxes = torch.Tensor(data['boxes'].tolist()),
                                                                       centers = torch.Tensor(data['centers'].tolist()),
                                                                       radius = torch.Tensor(data['radius'].tolist()),
                                                                       dirs = torch.Tensor(data['directions'].tolist()),
                                                                       labels = torch.Tensor(data['labels'].tolist())
                                                                    ))]
  
        prediction = [self.transform_for_metric(item) for item in predicted_items]

        # mAP metric evaluation for epoch over all validation data
        precision, recall = self.ME.evaluate(prediction,
                                             target,
                                             self.model.classes_ids,
                                             self.cfg.get("overlaps", [0.5]))

        print("")
        print(f' {" ": <9} "==== Precision ==== Recall ==== F1 ====" ')
       
        precision = np.mean(precision[:, -1])
        recall = np.mean(recall[:, -1])
        f1 = 2*precision*recall/(precision+recall)

        print("Overall_precision: {:.2f}".format(precision))
        print("Overall_recall: {:.2f}".format(recall ))
        print("F1: {:.2f}".format(f1))

        img = data['image']
        h,w = img.shape[:2]

        for box, center, radius, dirn in zip(data['boxes'], data['centers'], data['radius'], data['directions']):

            box[0::2] = w*box[0::2].clip(min=0, max=w)
            box[1::2] = h*box[1::2].clip(min=0, max=h)
            center[0] = w*center[0].clip(min=0, max=w)
            center[1] = h*center[1].clip(min=0, max=h)
            radius = radius*min(w,h)

            box = box.astype(np.int32)
            center = center.astype(np.int32)
            radius = radius.astype(np.int32)
            
            endpoint = center.copy()
            endpoint[1] += radius

            dirn -= np.pi/2

            # Direction point
            Cos = np.cos(np.deg2rad(0))
            Sin = np.sin(np.deg2rad(0))
            u = np.cos(dirn)*radius
            v = np.sin(dirn)*radius
            x = int(u*Sin - v*Cos + center[0])
            y = int(u*Cos - v*Sin + center[1])

            img = cv2.rectangle(img,(box[0], box[1]),(box[2], box[3]),(255,0,0),1)
            img = cv2.circle(img, center, 0, color=(0,255,0), thickness=3)
            img = cv2.line(img, center, endpoint, color = (255,255,0), thickness=1)
            img = cv2.circle(img, (x,y), 0, color=(0,255,255), thickness=3)

        for item in predicted_items[0]:

            box = item['box']
            center = item['center']
            radius = item['radius']
            dirn = item['direction']

            box[0::2] = w*box[0::2].clamp(min=0, max=w)
            box[1::2] = h*box[1::2].clamp(min=0, max=h)
            center[0] = w*center[0].clamp(min=0, max=w)
            center[1] = h*center[1].clamp(min=0, max=h)
            radius = radius*min(w,h)
            
            box = box.detach().cpu().numpy().astype(np.int32)
            center = center.detach().cpu().numpy().astype(np.int32)
            radius = radius.detach().cpu().numpy().astype(np.int32)
            dirn = dirn.detach().cpu().numpy().astype(np.float32)
            
            endpoint = center.copy()
            endpoint[0] += radius

            # Direction point
            dirn -= np.pi/2
            Cos = np.cos(np.deg2rad(0))
            Sin = np.sin(np.deg2rad(0))
            u = np.cos(dirn)*radius
            v = np.sin(dirn)*radius
            x = int(u*Sin - v*Cos + center[0])
            y = int(u*Cos - v*Sin + center[1])

            img = cv2.rectangle(img,(box[0], box[1]),(box[2], box[3]),(0,0,255),1)
            img = cv2.circle(img, center, 0, color=(255,0,255), thickness=3)
            img = cv2.line(img, center, endpoint, color = (60,150,255), thickness=1)
            img = cv2.circle(img, (x,y), 0, color=(177, 206, 251), thickness=3)


        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("image", 1300, 1600)
        cv2.imshow('image', img) 
        cv2.waitKey(0) 
        cv2.destroyAllWindows() 

    def run_testing(self):
        """Run test with test data split, computes mean average precision of the
        prediction results.
        """

        test_folder = self.cfg.log_dir + "test/"
        make_dir(test_folder)

        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

        log.info("DEVICE : {}".format(self.device))
        log_file_path = join(test_folder, 'log_test_' + timestamp + '.txt')
        log.info("Logging in file : {}".format(log_file_path))
        log.addHandler(logging.FileHandler(log_file_path))

        batcher = ConcatBatcher(self.device)

        test_split = TorchDataloader(dataset=self.dataset.get_split('testing'),
                                     preprocess=self.model.preprocess,
                                     transform=self.model.transform,
                                     use_cache=False,
                                     shuffle=False)

        testing_loader = DataLoader(
            test_split,
            batch_size=1,
            num_workers=self.cfg.get('num_workers', 4),
            pin_memory=self.cfg.get('pin_memory', False),
            collate_fn=batcher.collate_fn,
            worker_init_fn=lambda x: np.random.seed(x + np.uint32(
                torch.utils.data.get_worker_info().seed)))

        self.load_ckpt()
        self.model.eval()

        log.info("Started testing")
        prediction = []
        target = []

        with torch.no_grad():
            for data in tqdm(testing_loader, desc='testing'):
            
                data.to(self.device)
                self.model.head.image_size = data.images.shape[-2:]

                results = self.model(data)
                boxes_batch = self.model.inference_end(results)

                target.extend([self.transform_for_metric(self.transform_input_batch(boxes, centers, radius, dirs, labels=labels)) 
                            for boxes, centers, radius, dirs, labels in zip(data.boxes, data.centers, 
                                                                            data.radius, data.directions, data.labels)])

                prediction.extend([self.transform_for_metric(boxes) for boxes in boxes_batch])

        # mAP metric evaluation for epoch over all validation data
        precision, recall = self.ME.evaluate(prediction,
                                             target,
                                             self.model.classes_ids,
                                             self.cfg.get("overlaps", [0.5]))
        
        log.info("")
        log.info(f' {" ": <9} "==== Precision ==== Recall ==== F1 ====" ')
        for i, c in enumerate(self.model.classes):

            p = precision[i,0]
            rec = recall[i,0]
            f1 = 2*p*rec/(p+rec)
            log.info(f' {"{}".format(c): <15} {"{:.2f}".format(p): <15.5} {"{:.2f}".format(rec): <10} {"{:.2f}".format(f1)}')

        precision = np.mean(precision[:, -1])
        recall = np.mean(recall[:, -1])
        f1 = 2*precision*recall/(precision+recall)

        log.info("")
        log.info("Overall_precision: {:.2f}".format(precision))
        log.info("Overall_recall: {:.2f}".format(recall ))
        log.info("F1: {:.2f}".format(f1))

        precision = float(precision)
        recall = float(recall)
        f1 = float(f1)
        
        test_protocol = {
            '0_model': self.cfg.get('model_name', None),
            '1_model_version':self.cfg.get('resume_from', None), 
            '2_dataset': self.cfg.get('dataset_name', None), 
            '3_date': datetime.now().strftime('%Y-%m-%d_%H:%M:%S'), 
            '4_precision': precision, 
            '5_recall': recall, 
            '6_f1': f1
                    }

        with open(test_folder + 'test_protocol.yaml', 'w') as outfile:
            yaml.dump(test_protocol, outfile)

    def run_validation(self, epoch=0):
        """Run validation with validation data split, computes mean average
        precision and the loss of the prediction results.
        Args:
            epoch (int): step for TensorBoard summary. Defaults to 0 if
                unspecified.
        """

        # Model in evaluation mode -> no gradient = parameters are not optimized
        self.model.eval()

        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

        batcher = ConcatBatcher(self.device)

        valid_dataset = self.dataset.get_split('validation')

        valid_split = TorchDataloader(dataset=valid_dataset,
                                      preprocess=self.model.preprocess,
                                      transform=self.model.transform,
                                      use_cache=self.dataset.cfg.use_cache,
                                      shuffle=False,
                                      steps_per_epoch=self.dataset.cfg.get(
                                          'steps_per_epoch_valid', None))

        validation_loader = DataLoader(
            valid_split,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.get('num_workers', 4),
            pin_memory=self.cfg.get('pin_memory', False),
            collate_fn=batcher.collate_fn,
            worker_init_fn=lambda x: np.random.seed(x + np.uint32(
                torch.utils.data.get_worker_info().seed)))

        log.info("Started validation")

        self.valid_losses = {}

        prediction = []
        target = []

        with torch.no_grad():
            for data in tqdm(validation_loader, desc='validation'):

                data.to(self.device)
                # If Pipeline should process batches of various image size, priors feature maps needs to adapt according
                # to this size. Head image_size parameter is updated with every batch, so detection head is able
                # to adapt for various image size. This update is performed here within a pipeline, when image batch is
                # formed for forward pass.
                #self.model.head.image_size = data.images.shape[-2:]

                results = self.model(data)
                loss = self.model.loss(results, data)

                for l, v in loss.items():
                    if l not in self.valid_losses:
                        self.valid_losses[l] = []
                        self.valid_losses[l].append(v.cpu().numpy())

                # convert to bboxes for mAP evaluation
                
                boxes_batch = self.model.inference_end(results)
                
                target.extend([self.transform_for_metric(self.transform_input_batch(boxes, centers, radius, dirs, labels=labels)) 
                               for boxes, centers, radius, dirs, labels in zip(data.boxes, data.centers, 
                                                                               data.radius, data.directions, data.labels)])



                prediction.extend([self.transform_for_metric(boxes) for boxes in boxes_batch])

        # Process bar data feed
        sum_loss = 0
        desc = "validation - "
        for l, v in self.valid_losses.items():
            desc += " %s: %.03f" % (l, np.mean(v))
            sum_loss += np.mean(v)

        desc += " > loss: %.03f" % sum_loss
        log.info(desc)

        # mAP metric evaluation for epoch over all validation data
        precision, recall = self.ME.evaluate(prediction,
                                             target,
                                             self.model.classes_ids,
                                             self.cfg.get("overlaps", [0.5]))

        log.info("")
        log.info(f' {" ": <9} "==== Precision ==== Recall ==== F1 ====" ')
        for i, c in enumerate(self.model.classes):

            p = precision[i,0]
            rec = recall[i,0]
            f1 = 2*p*rec/(p+rec)
            log.info(f' {"{}".format(c): <15} {"{:.2f}".format(p): <15.5} {"{:.2f}".format(rec): <10} {"{:.2f}".format(f1)}')

        precision = np.mean(precision[:, -1])
        recall = np.mean(recall[:, -1])
        f1 = 2*precision*recall/(precision+recall)

        log.info("")
        log.info("Overall_precision: {:.2f}".format(precision))
        log.info("Overall_recall: {:.2f}".format(recall ))
        log.info("F1: {:.2f}".format(f1))
        
        self.valid_losses["precision"] = precision
        self.valid_losses["recall"] = recall
        self.valid_losses["f1"] = f1

        return self.valid_losses


    def run_training(self):

        if not os.path.exists(self.cfg.log_dir + 'process_config.json'):
            with open(self.cfg.log_dir + 'process_config.json', "w") as outfile:
                json.dump(dict(self.global_cfg), outfile)

        """Run training with train data split."""
        torch.manual_seed(self.rng.integers(np.iinfo(np.int32).max))  # Random reproducible seed for torch

        log.info("DEVICE : {}".format(self.device))
        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

        log_file_path = join(self.cfg.log_dir, 'log_train_' + timestamp + '.txt')
        log.info("Logging in file : {}".format(log_file_path))
        log.addHandler(logging.FileHandler(log_file_path))

        batcher = ConcatBatcher(self.device)

        train_dataset = self.dataset.get_split('training')

        train_split = TorchDataloader(dataset=train_dataset,
                                      preprocess=self.model.preprocess,
                                      transform=self.model.transform,
                                      use_cache=self.dataset.cfg.use_cache,
                                      steps_per_epoch=self.dataset.cfg.get(
                                          'steps_per_epoch_train', None))

        self.optimizer, self.scheduler = self.model.get_optimizer(self.cfg.optimizer)

        start_ep, _ = self.load_ckpt()

        epoch_size = math.ceil(len(train_dataset) / self.cfg.batch_size)
        stepvalues = (self.cfg.decay1 * epoch_size, self.cfg.decay2 * epoch_size)
        step_index = 0
        iteration = 0

        if os.path.exists(self.cfg.log_dir + '/training_record.csv'):
            training_record = pd.read_csv(self.cfg.log_dir + '/training_record.csv', index_col=False)
        else:
            training_record = pd.DataFrame([],columns=['epoch', 'precision', 'recall', 'f1', 
                                                       'class_loss', 'box_loss', 'center_loss',
                                                       'radius_loss','direction_loss'])

        log.info("Started training")
        
        for epoch in range(start_ep+1, self.cfg.max_epoch + 1):

            log.info(f'================================ EPOCH {epoch:d}/{self.cfg.max_epoch:d} ================================')
            self.model.train()
            self.losses = {}
            #print(sum(p.numel() for p in self.model.parameters() if p.requires_grad))

            train_loader = DataLoader(
            train_split,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.get('num_workers', 4),
            pin_memory=self.cfg.get('pin_memory', False),
            collate_fn=batcher.collate_fn,
            worker_init_fn=lambda x: np.random.seed(x + np.uint32(
                torch.utils.data.get_worker_info().seed))
                                    )
            
            #if iteration in stepvalues:
            #    step_index += 1

            process_bar = tqdm(train_loader, desc='training')

            for data in process_bar:

                iteration += 1
                if iteration in stepvalues:
                    step_index += 1

                #lr = adjust_learning_rate(self.optimizer, self.cfg.optimizer.gamma, epoch,
                #                          step_index, iteration, epoch_size)

                #img = data.images[0].squeeze(0).permute(1,2,0).cpu().detach().numpy() 
                #cv2.namedWindow("image", cv2.WINDOW_NORMAL)
                #cv2.resizeWindow("image", 1300, 1600)
                #cv2.imshow('image', img) 
                #cv2.waitKey(0) 
                #cv2.destroyAllWindows() 
                #print(data.centers)
                data.to(self.device)

                # If Pipeline should process batches of various image size, priors feature maps needs to adapt according
                # to this size. Head image_size parameter is updated with every batch, so detection head is able
                # to adapt for various image size. This update is performed here within a pipeline, when image batch is
                # formed for forward pass.

                #self.model.head.image_size = data.images.shape[-2:]
                results = self.model(data)

                loss = self.model.loss(results, data)

                loss_sum = sum([self.cfg.loss.cls_weight * loss['loss_cls'],
                                self.cfg.loss.loc_weight * loss['loss_box'],
                                self.cfg.loss.center_weight * loss['loss_center'],
                                self.cfg.loss.radius_weight * loss['loss_radius'],
                                self.cfg.loss.dir_weight * loss['loss_dir']
                               ]) 
 
                self.optimizer.zero_grad()
                loss_sum.backward()
                
                #if self.cfg.get('grad_clip_norm', -1) > 0:
                #    torch.nn.utils.clip_grad_value_(self.model.parameters(),
                #                                    self.cfg.grad_clip_norm)
                
                self.optimizer.step()

                desc = "training - "
                for l, v in loss.items():
                    if l not in self.losses:
                        self.losses[l] = []
                    self.losses[l].append(v.cpu().detach().numpy())
                    desc += " %s: %.03f" % (l, v.cpu().detach().numpy())

                desc += " > loss: %.03f" % loss_sum.cpu().detach().numpy()
                process_bar.set_description(desc)
                process_bar.refresh()

        #    # Scheduler is defined in model.get_oprimizer -> by default there is None value right now for scheduler
        #    if self.scheduler is not None:
        #        self.scheduler.step()

            if os.path.exists(self.cfg.log_dir + '/metrics.npy'):
                metrics = np.load(self.cfg.log_dir + '/metrics.npy')
                best_f1 = metrics[2]
            else:
                best_f1 = 0
            # --------------------- validation of epoch -> given by self.run_valid()
            if (epoch % self.cfg.get("validation_freq", 1)) == 0:

                metrics = self.run_validation()

                training_record.loc[epoch] = [epoch, metrics['precision'], metrics['recall'], metrics['f1'],
                                              metrics['loss_cls'][0], metrics['loss_box'][0], metrics['loss_center'][0],
                                              metrics['loss_radius'][0],metrics['loss_dir'][0]]


                actual_f1 = metrics['f1']
                
                if actual_f1 > best_f1:

                    best_f1 = actual_f1
                    self.save_ckpt(epoch, save_best=True)
                    np.save(self.cfg.log_dir + '/metrics.npy', 
                            np.array([metrics['precision'], metrics['recall'], metrics['f1'],
                                      metrics['loss_cls'][0], metrics['loss_box'][0], metrics['loss_center'][0],
                                      metrics['loss_radius'][0], metrics['loss_dir'][0]]))
                
            if epoch % self.cfg.save_ckpt_freq == 0:
                self.save_ckpt(epoch,save_best=False)

            training_record.to_csv(self.cfg.log_dir + '/training_record.csv', index=False)