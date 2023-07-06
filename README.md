<h1>Detection of C-shapes</h1>

<h2>Introduction</h2>
The code in this project/repository is designed for training and experimenting with RetinaNet object detection model. Framework expects two-dimensional RGB image as a source data. Framework is configured to train model detecting C-shaped objects. Dataset prepared for use by this framework is located in <b>smb://oelnas.riegl-gmbh/scandaten/AI_DATA/Object_Detection/2D/Cshapes/2023-03-22_11-47-02/</b>.

<h2>Installation</h2>

<h3>Requirements</h3>

- Installed Anaconda/Miniconda

- Essential python packages:
  - [pytorch==2.0.1](https://pytorch.org/get-started/)
  - [torchvision==0.15.2](https://pytorch.org/vision/stable/index.html)
  - [opencv==4.8.0.74 ](https://pypi.org/project/opencv-python/)
  - [numpy==1.24.4](https://numpy.org/)
  - [numba==0.57.1](https://numba.pydata.org/)

<h3>Local setup</h3>

```
1. open Anaconda Prompt

2. Download projet to your local machine
 ```git clone git@gitlab.riegl-gmbh:xp/RetinaC.git```

3. Go to the project folder:
cd /Path/To/RetinaC/

4. Create and activate virtual environment
conda create --name "env_name" python=3.10
conda activate "env_name"

5. Install dependecies with requirements. txt file
pip3 install -r requirements.txt
```

<h2>Annotated dataset simulation</h2>
Simulated dataset was used to train c-shapes detection model. Dataset simulator generating binary images with specific geometrical shapes was developed for this purpose. Target (C-shapes) objects in the images are partially occluded circles and other objects (lines, clusters of noise pixels) are added as a noise.

Data simulation module can be found in **DataSimulator** folder.

<h3>Structure of module</h3>

-   <b>augmentation.py</b> – Generates in the image noisy pixels and noisy geometrical shapes.
-	<b>bresenham.py</b> – Algorithm for line generation in the image.
-	<b>ops.py</b> – Collection of functions used in data simulation process.
-	<b>shapes.py</b> – Classes defining geometrical shapes, which are created in simulation.
-   <b>simulator.py</b> - Script which starts simulation process and generates simulated dataset with given configuration.

<h3>Configuration file for data simulation process <b>config_sim.yaml</b></h3>

-   general – General simulation parameters.
    -   output_path - Directory where folder with simulated dataset should be created.
    -   sample_size - Number of generated images (dataset size).
    -   sample_split -  Defines proportion of dataset split in order training/validation/testing.
    -   pixel_limit - Range (in pixels) used for random selection, defining bigger image dimension.
    -   img_aspect_ratio - Range (in pixels) used for random selection, defining image aspect ratio.
-	oval_simulator – Parameters defining simulated target (C-shapes) attributes.
    -   radius_limit - Range (in pixels) used for random selection, defining target radius.
    -   ellipse_aspect_ratio - Range used for random selection, defining ellipse aspect ratio.
    -   circle_prob – Probability of Circle-Ellipse target simulation (binary selection).
    -   line_thickness - Range (in pixels) used for random selection, defining thickness of line of simulated target, which is drawn in image by cv2ellipse() function.
    -   occlusion – Parameters of simulated target occlusion.
        -   prob - Probability of occlusion.
        -   range - Range used for random selection, defining how big part of target is occluded.
-	augmentation – Parameters used for noise generation in simulated image.
    -	border_distortion – Augmentation decreasing intensity value of pixels of simulated geometrical shapes.
        -   prob – Probability of application of this augmentation technique.
        -   intensity_range - Range used for random selection, defining minimal intensity decrement level.
    -   blur – Parameters used for image blurring.
        -   prob -  Probability of application of blurring augmentation.
        -   kernel_size - Range (in pixels) used for random selection, defining blurring kernel size.
    -   noise – Parameters used for simulation of noisy geometrical objects.
        -   type_prob – Probability of (linear, cluster) geometrical shape.
        -   linear – Parameters used for linear geometrical object simulation.
            -   line_range - Range (in pixels) used for random selection, defining linear geometrical object length.
            -   corner_prob – Probability if simulated linear object will be in the shape of corner (two connected lines) or simple line.
        -   cluster - Parameters used for random circular noise pixel cluster simulation.
            -   rad_range - Range (in pixels) used for random selection, defining radius of circular noise pixel cluster.

<h2>Execution of framework sessions</h2>

```
conda activate "env_name"
```

1. Run training session
```
python run_training.py
```

2. Run single sample inference session with visualization
```
python show_inference.py
```

3. Run model performance evaluation on testing dataset
```
python run_testing.py
```

<h2>Framework structure</h2>

-	<b>dataset</b> – contains modules for data preprocessing and importing data to Pytorch framework for training/validation/testing session
    -	<b>base_dataset.py</b> - Base dataset class, all datasets must inherit from this class to be compatible with pipeline. 
    -	<b>ImageDataset.py</b> - Imports data for given data split (training/validation/testing) and defines how single sample (image+annotation) should be imported.
    -	<b>dataloaders.py</b> – Custom dataloader which forms images and annotations into batches as an input for model.
-	<b>augment</b> – module is responsible for augmentation applied on single image during training process.
    -	<b>augmentation.py</b> – Class implementing set of custom image augmentation techniques, which (are/can be) applied during training process.
    -	<b>ops.py</b> -  Collection of operations used for image augmentation.
-	model – Contains modules defining RetinaNet structure and all methods for network training and inference.
    -	<b>base_model.py</b> – Base model class, all models must inherit from this class to be compatible with pipeline.
    -	<b>MobileNet.py</b> – Architecture of MobileNetV1 classifier (used as backbone of RetinaNet).
    -	<b>RetinaNet.py</b> – Architecture of RetinaNet detector with methods for training and inference.
    -	<b>utils.py</b> – Collection of function for target encoding/decoding, anchor generator and NMS postprocessing.
-	<b>pipeline</b> – Object detection pipeline defines process of training, testing and model prediction visualization.
    -	<b>base_pipeline.py</b> -  Base pipeline class, all pipelines must inherit from this class to be compatible with pipeline. Determines log directory of training/inference session.
    -	<b>pipeline.py</b> – Object Detection pipeline defines logic of training, validation, testing, inference session. It uses dataset  module to introduce data to model, performs required actions and formats model output.
    -	<b>metrics.py</b> – Module evaluates precision and recall based on model prediction and ground truth for given data batch.
    -	<b>utils.py</b> - Collection of function used in Object Detection pipeline.
-	<b>weights</b> – Contains mobilenet.tar file of pretrained weights for MobileNet backbone.
-	<b>config.py</b> – Config class in this module is responsible for correct parsing of config.yaml which contains dataset, model, pipeline parameters
-	<b>run_training.py</b> – Starts training session (new one or resumed one).
-	<b>run_testing.py</b> – Evaluates performance of trained model on testing dataset.
-	<b>show_inference.py</b> – Performs prediction with trained model on single image and visualize it.

<h2>Framework configuration file <b>config.yaml</b></h2>

-	<b>global_args</b> – parameters shared globally through the whole framework
    -	<b>device</b> - Device used by framework (cpu or cuda)
    -	<b>seed</b> - The seed is used to initialize the random number generator, to control reproducibility of model training.
    -	<b>output_path</b> - Directory, where pipeline output will be stored.
-	<b>dataset</b> – parameters used in dataset  module
    -	<b>dataset_path</b> - Directory, where training/validation/testing split folders are located.
-	<b>model</b> - parameters used in model module
    -	<b>model_name</b> - Name of used model.
    -	<b>backbone</b> - Defines which backbone architecture should be used (mobilenet or resnet)
    -	<b>pretrained</b> - Defines if model training should start with already pretrained weights of backbone or it will be randomly initialized.
    -	<b>classes</b> - list of object names (strings), defining which categories model detects
    -	<b>mobilenet</b> - Parameters used for training and validation for model with MobileNet backbone architecture.
        -	<b>batch_size</b> - Batch size used by dataloader.
        -	<b>image_size</b> – Image size used in preprocessing/augmentation stage.
        -	<b>return_layers</b> – Dictionary defining which items of backbone dictionary are returned as feature maps to Retina network. There are 3 feature maps.
        -	<b>in_channel</b>: Input channel number of backbone.
        -	<b>out_channel</b>: Output channel number of backbone.
    - <b>resnet</b> - Parameters used for training and validation for model with Resnet50 backbone architecture.
        -	<b>batch_size</b> - Batch size used by dataloader.
        -	<b>image_size</b> – Image size used in preprocessing/augmentation stage.
        -	<b>return_layers</b> – Dictionary defining which items of backbone dictionary are returned as feature maps to Retina network. There are 3 feature maps.
        -	<b>in_channel</b>: Input channel number of backbone.
        -	<b>out_channel</b>: Output channel number of backbone.
    - <b>head</b> – Parameters defining shape of anchors used for detection, thresholds for NMS and inference procedure.  
        -	<b>steps</b> - spacing/stride (in pixels) of anchors for corresponding feature maps used for anchors generation
        -	<b>aspects</b> – list of anchor aspect ratios used for anchors generation, it is shared for all feature maps  
        -	<b>sizes</b> – list of sizes (in pixels) of anchors for corresponding feature maps used for anchors generation
        -	<b>variances</b> – normalizing factors used for encoding/decoding between anchors and ground truth targets
        -	<b>negpos_ratio</b> - Defines how many negative samples is selected for 1 positive sample and introduced to loss function during training procedure.
        -	<b>nms_top_k</b> -  Defines K anchors with confidence score, which are considered as model prediction.
        -	<b>nms_thresh</b> – Maximal allowed IOU between predicted bounding boxes. If IOU is higher, bounding box with higher confidence score is kept for prediction. 
        - <b>score_thresh</b> – Minimal confidence score of anchor to be considered for prediction. Anchors with lower score can’t be predicted by model.
        - <b>Iou_thr</b>– Defines IOU threshold between predicted bounding box and ground truth bounding box, which distinguish between positive and negative matches. If anchor has IOU at least on this level with any ground truth bounding box, than it is considered as positive sample.
-	<b>augment</b> – Dictionary which defines parameters used in augmentation module.
-	<b>pipeline</b> – parameters used in pipeline module
    -	<b>inference_mode</b> (True/False) – Determines how is the model initialized for given session. Should be False for training session and True for any inference session.
    -	<b>is_resume</b> – Determines if training session should initialize new training session (False) or to continue from previously initialized training session (True)
    -	<b>resume_from</b> – Defines which training session is used for inference or resumed training. It is name of folder in form of timestamp, which was generated by pipeline during training session. Checkpoint having best performance (f1) score on validation dataset is used for inference or resumed training.
    -	<b>max_epoch</b> – defines maximal number of training epochs performed during given training session
    -	<b>validation_freq</b> – Defines frequency of validation session execution during training session. Validation is executed after given number of training epochs.
    -	<b>save_ckpt_freq</b> - Defines frequency of exporting model state. Export is executed after given number of training epochs.
    -	<b>num_workers</b> – number of workers used for parallel data batch import in Dataloader
    -	<b>overlaps</b> – IOU threshold (can be specific for each class for multiclass detection) used for Precision and Recall evaluation. Determines if predicted bounding box is True or False positive.
        -	<b>loss</b> – Parameters used for loss computation.
            -	<b>cls_weight</b> – normalizing factor for classification loss
            -	<b>loc_weight</b> – normalizing factor for bounding box regression loss
            -	<b>center_weight</b> – normalizing factor for C-shape center loss
            -	<b>dir_weight</b> – normalizing factor for C-shape direction loss
        -	<b>optimizer</b> – Parametrs of SGD optimizer.  
            -	<b>lr</b> – learning rate
            -	<b>momentum</b>
            -	<b>weight_decay</b>
