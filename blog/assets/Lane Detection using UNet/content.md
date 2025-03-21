# Lane Detection using UNet

Published Date: March 18, 2025
Tags: Computer Vision, Featured, Machine Learning

I have been working on road lane detection using LaneNet by TensorFlow 2.x for the past two months. In the beginning, I chose to build a road lane detection program by using the Canny Edge method utilizing the contrast feature of images as the base of the detection method. The result was not quite good. It was not adaptable to various road conditions, such as curvature, terrain, brightness, etc. In order to improve it, I decided to do further research about methods that are more robust. Later, I found that using Image Segmentation could make the system more adaptable to road conditions. One of the research papers that I found was “Towards End-to-End Lane Detection: an Instance Segmentation Approach” by Davy Neven et. al. ([https://arxiv.org/abs/1802.05591](https://arxiv.org/abs/1802.05591))

In a nutshell, what this model, LaneNet, does is detect lane or non-lane pixels using binary segmentation as the base detection method. And what makes it perform better from contrast-based detection is the ability to segment the area of the lane and non-lane area. In addition to that, the LaneNet model could segment each lane without using multi-categorical segmentation.

To implement the model, I used CARLA as a driving simulator for testing the model, and also to extract the image for datasets. CARLA is an open-source simulator for autonomous driving research that allows us to modify the map, weather, and surrounding condition.

For this project, I used CARLA 0.9.10. However, I am not sure about the other versions if they would work or not.

Here are some of the libraries that I used:

- Tensorflow-gpu 2.7.0
- Numpy 1.17.4
- opencv-python 4.5.4
- pandas 0.25.3
- Pillow 8.4.0
- random
- scikit-learn 0.24.1
- Tensorboard

The overview of the schematic is shown in Fig 1. There are two branches of segmentation, they are binary segmentation and instance segmentation. The binary segmentation branch is simply detecting the lane or non-lane area of each pixel on the RGB input image. The main role of instance segmentation is to segment the area of the image in different colors on each lane. To get the results, both branch outputs are multiplied by each other. Finally, to make the lanes look smoother, H-Net is added to the last layer.

![Fig 1. LaneNet Architecture](https://miro.medium.com/v2/resize:fit:1225/1*GeX_xepllG_GeQqk6JMVOQ.png)

Fig 1. LaneNet Architecture

In this post, I will explain how I build the datasets, the model, and train the program, what configuration I choose for the loss function, and the result of testing it with CARLA.

# **Dataset**

### Extracting Images from CARLA

Building datasets is probably the most bothersome part of the implementation process. I will walk you through it with some examples that have been done by some contributors to make it simpler.

Since labeling datasets can be a real hassle, I used a dataset extractor for road lane detection in CARLA, which was built by Github user, Glutamat42 ([https://github.com/Glutamat42/Carla-Lane-Detection-Dataset-Generation](https://github.com/Glutamat42/Carla-Lane-Detection-Dataset-Generation)). Check out his Github Repository to understand it better. Though, I modified some of the codes for it to work on my project. Here, I will explain the modified codes.

```python
def __init__(self):
    if(cfg.isThirdPerson):
        self.camera_transforms = [carla.Transform(carla.Location(x=-4.5, z=2.2), carla.Rotation(pitch=-14.5)),
                                  carla.Transform(carla.Location(x=-4.0, z=2.2), carla.Rotation(pitch=-18.0))]
    else:
        # self.camera_transforms = [carla.Transform(carla.Location(x=0.0, z=3.2), carla.Rotation(pitch=-19.5)), # camera 1
        #                           carla.Transform(carla.Location(x=0.0, z=2.8), carla.Rotation(pitch=-18.5)), # camera 2
        #                           carla.Transform(carla.Location(x=0.3, z=2.4), carla.Rotation(pitch=-15.0)), # camera 3
        #                           carla.Transform(carla.Location(x=1.1, z=2.0), carla.Rotation(pitch=-16.5)), # camera 4
        #                           carla.Transform(carla.Location(x=1.0, z=2.0), carla.Rotation(pitch=-18.5)), # camera 5
        #                           carla.Transform(carla.Location(x=1.4, z=1.2), carla.Rotation(pitch=-13.5)), # camera 6
        #                           carla.Transform(carla.Location(x=1.8, z=1.2), carla.Rotation(pitch=-14.5)), # camera 7
        #                           carla.Transform(carla.Location(x=2.17, z=0.9), carla.Rotation(pitch=-14.5)), # camera 8
        #                           carla.Transform(carla.Location(x=2.2, z=0.7), carla.Rotation(pitch=-11.5))] # camera 9
        self.camera_transforms = [carla.Transform(carla.Location(x=1.6, z=1.7))]
    

def reset_vehicle_position(self):
    #camera_index = random.randint(0,len(self.camera_transforms)-1)
    camera_index = 0
    
    self.camera_rgb.set_transform(self.camera_transforms[camera_index])
    self.camera_semseg.set_transform(self.camera_transforms[camera_index])
    print("Camera Index: ", camera_index)
```

The fast_lane_detection.py file runs the vehicle simulation in CARLA with predetermined conditions for the front camera, weather, and maps in the file itself. This program also detects the road lanes using the CarlaLane2D function. However, the creator of this program harvests datasets with variations on the camera position but only one condition for the weather. Because I only planned to implement it with one weather condition and a fixed camera position, I modified the program to only have one camera position and be able to adjust the weather conditions.

Running the fast_lane_detection.py program creates the .npy files and two folders named “data” to store the output, and “debug” to check if there are any broken files (for example, if the program is not able to detect the lanes properly). After checking if everything is good to go, executing the ‘dataset_generator.py’ file will export the original images and the JSON file, which has lane positions in the picture.

### Creating the Masks

After the JSON file and original images are exported, I built the masks for binary segmentation and instance segmentation by the information of the pixel position. Here are the codes for how to build the masks.

```python
import argparse
import glob
import json
import os
import os.path as ops
import shutil

import cv2
import numpy as np

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, help='The origin path of unzipped tusimple dataset')
    return parser.parse_args()

def process_tusimple_dataset(src_dir):
    traing_folder_path = ops.join(ops.split(src_dir)[0], 'training')
    os.makedirs(traing_folder_path, exist_ok=True)

    gt_image_dir = ops.join(traing_folder_path, 'gt_image')
    gt_binary_dir = ops.join(traing_folder_path, 'gt_binary_image')
    gt_instance_dir = ops.join(traing_folder_path, 'gt_instance_image')

    os.makedirs(gt_image_dir, exist_ok=True)
    os.makedirs(gt_binary_dir, exist_ok=True)
    os.makedirs(gt_instance_dir, exist_ok=True)

    for idx, json_label_path in enumerate(glob.glob('{:s}/*.json'.format(src_dir))):
        get_image_to_folders(json_label_path, gt_image_dir, gt_binary_dir, gt_instance_dir, src_dir, idx)
    #gen_train_sample(src_dir, gt_binary_dir, gt_instance_dir, gt_image_dir)
    #split_train_txt(src_dir)

def get_image_to_folders(json_label_path, gt_image_dir, gt_binary_dir, gt_instance_dir, src_dir, idx):
    json_gt = [json.loads(line) for line in open(json_label_path)]
    for id, gt in enumerate(json_gt):
        filename = '{}_{}.png'.format(idx, id)
        gt_lanes = gt['lanes']
        y_samples = gt['h_samples']
        raw_file = gt['raw_file']

        raw_img = cv2.imread(raw_file)
        
        gt_lanes_vis = [[(x,y) for (x,y) in zip(lane, y_samples) if x >=0] for lane in gt_lanes]
        img_vis = raw_img

        bin_background = np.zeros_like(raw_img)
        inst_background = np.zeros_like(raw_img)
        skip=False

        #for lane in gt_lanes_vis:
            

        colors = [[70,70,70], [120,120,120], [20,20,20], [170,170,170]]
        color_bw = [[255,255,255], [255,255,255], [255,255,255], [255,255,255]]
        for i in range(len(gt_lanes_vis)):
            if len(gt_lanes_vis[i]) == 0:
                skip=False
                break
            else: 
                inst_mask = cv2.polylines(inst_background, np.int32([gt_lanes_vis[i]]), isClosed=False, color=colors[i], thickness=5)
                bin_mask = cv2.polylines(bin_background, np.int32([gt_lanes_vis[i]]), isClosed=False, color=color_bw[i], thickness=5)
                skip=False

        if skip==True:
            print('Number of lanes is not 4!')
        else:
            shutil.copy(raw_file, ops.join(gt_image_dir, filename))
            cv2.imwrite(ops.join(gt_binary_dir, filename), bin_mask)
            cv2.imwrite(ops.join(gt_instance_dir, filename), inst_mask)

if __name__ == '__main__':
    args = init_args()
    process_tusimple_dataset(args.src_dir)

json_label_path = './data/dataset/Town03/train_gt.json'

```

After the masks are built, the images are supposed to look like the following images.

![](https://miro.medium.com/v2/resize:fit:1225/1*Kfm2Z777W_nqGDJkvkQkkg.png)

### Processing the Datasets

For the input images, it will have to be resized as the input dimension of the model is(512, 256). The code for preprocessing the input images is shown below.

```python
def read_txt(root, flag):
    img_path = []
    bin_path = []
    inst_path = []

    train_file = ops.join(root, 'train.txt')
    val_file = ops.join(root, 'val.txt')
    test_file = ops.join(root, 'test.txt')

    if flag == 'train':
        assert exists(train_file)
        file_open = train_file
    elif flag == 'valid':
        assert exists(val_file)
        file_open = val_file
    else:
        assert exists(test_file)
        file_open = test_file

    df = pd.read_csv(file_open, header=None, delim_whitespace=True, names=['img', 'bin', 'inst'])
    #print(df)
    img_path = df['img'].values
    bin_path = df['bin'].values
    inst_path = df['inst'].values

    #print(img_path)
    return img_path, bin_path, inst_path

def preprocessing(img_path, bin_path, inst_path, resize=(512,256))
    image_ds = []
    for i, image_name in enumerate(img_path):
        image = cv2.imread(image_name)
        image = Image.fromarray(image)
        image = image.resize(resize)
        image_ds.append(np.array(image, dtype=np.float32))
     
    mask_ds = []
    for i, image_name in enumerate(bin_path):
        image = cv2.imread(image_name, 0)
        image = Image.fromarray(image)
        image = image.resize(resize)
        label_binary = np.zeros([resize[1], resize[0]], dtype=np.uint8)
        mask = np.where(np.array(image)[:,:] != [0])
        #print(np.unique(np.array(image)))
        label_binary[mask] = 1
        mask_ds.append(np.array(label_binary, dtype=np.uint8))
        
    inst_ds = []
    for i, image_name in enumerate(inst_path):
        image = cv2.imread(image_name, 0)
        image = Image.fromarray(image)
        image = image.resize(resize)
        #print(np.unique(np.array(image)))
        inst_ds.append(np.array(ex,dtype=np.float32))
        
    return image_ds, bin_ds, inst_ds
```

To preprocess the binary masks, I resized the binary masks into the same size as (512, 256). However, some resizing methods might make the color of the pixels change. Because each pixel will be used later as a binary mask, which has to be either 1 or 0, the changing color after the resizing process could make the pixels not only 1 or 0 but also in the between. So what I did was resize it with some modifications on the following codes.

```python
def GRResize(im, size, filter):
    # Convert to numpy array of float
    arr = np.array(im, dtype=np.float32) / 255.0
    # Convert sRGB -> linear
    arr = np.where(arr <= 0.04045, arr/12.92, ((arr+0.055)/1.055)**2.4)
    # Resize using PIL
    arrOut = np.zeros((size[1], size[0]))
    chan = Image.fromarray(arr[:,:,i])
    chan = chan.resize(size, filter)
    arrOut[:,:] = np.array(chan).clip(0.0, 1.0)
    # Convert linear -> sRGB
    arrOut = np.where(arrOut <= 0.0031308, 12.92*arrOut, 1.055*arrOut**(1.0/2.4) - 0.055)
    # Convert to 8-bit
    arrOut = np.uint8(np.rint(arrOut * 255.0))
    # Convert back to PIL
    return Image.fromarray(arrOut)
```

The same with processing the binary masks, I resized it with the same modified resizing method for the instance masks.

After I preprocessed the datasets, I split the datasets with a ratio of 85:15 for training and testing data respectively. I used 6891 data for training datasets, 1216 data for validation/testing data. Here is the code for splitting the datasets into training and testing datasets.

Dataset Splitting Function

## Model Architecture

In this model that I built in this project, there are some modifications needed in order to ease the implementation. In the research paper, the architecture model that is used is ENet. For simplicity’s sake, I used UNet instead. Based on the paper by Alexander Karimov et.al ([https://arxiv.org/abs/1909.06840](https://arxiv.org/abs/1909.06840)), in the case of semantic segmentation, ENet performs computationally faster by 8–15 times than UNet. Although, it can perform less accurately than UNet by 1–2%. Furthermore, I also did not implement H-Net to the model.

The configuration of UNet itself is shown in Fig 2. The encoder of U-Net has shared networks for both binary and instance segmentation. Each decoder is linked with their respective branches, one for binary segmentation, and the other for instance segmentation.

![FIg 2. U-Net Architecture](https://miro.medium.com/v2/resize:fit:1225/0*wRwkJFCXVxpokYGP.png)

FIg 2. U-Net Architecture

The input that I used for this model was 512x256 RGB images with 3 channels, which makes the input dimension of the model to be [Batch Size, 256, 512, 3]. The output of the binary segmentation is [Batch Size, 256, 512, 1] while the instance segmentation output is [Batch Size, 256, 512, 4].

U-Net Model

## Loss Function

To calculate the loss function for binary segmentation, I decided to use the Focal Loss Function. This method was chosen to address the problem of the existing binary mask having imbalanced labels between lane and non-lane pixel labels.

For the instance segmentation, I used the Discriminative Loss as what is written in the paper. Discriminative Loss was originally researched by Bert de Brabandere, et. al. ([https://arxiv.org/abs/1708.02551](https://arxiv.org/abs/1708.02551)). The idea of it is to separate the objects of the same class in an image by the inter-cluster push force and intra-cluster pull force.

Here are the codes for the Discriminative Loss, which I cited from ([https://github.com/hq-jiang/instance-segmentation-with-discriminative-loss-tensorflow](https://github.com/hq-jiang/instance-segmentation-with-discriminative-loss-tensorflow)).

Discriminative Loss Function

For the metric function of the binary segmentation, I chose to compute the mean IoU (Interest of Union). The reason I did not choose accuracy for the metric was that accuracy is only computed if the predicted pixel is registered in a correct class or not. Since the label from most of the pixels in one image is non-lane, the accuracy will be high or at least around 70% even at the beginning of the training process, which is not an accurate representation if the model is well-trained or not. This is where the mean IoU method becomes useful as it only calculates the true positive among the false positive, true negative, and false negative, which is more suitable and appropriate for evaluating if the model is accurate enough.

IoU= true_positive / (true_positive + false_positive + false_negative).

Mean IoU Function

# **TRAINING**

For this training method, I chose to use the Adam optimizer as the optimizer algorithm, with a learning rate of 1e-4, weight decay of 1e-6, number batch size of 15, number of epochs at 100, and some callbacks functions:

1. Terminate on NaN

2. Early stopping when the validation loss is increasing for 4 epochs

3. Checkpoint that always saves the weight of every epoch.

Training Process

I trained it with NVIDIA Tesla P40 24GB for about 1.5 hours. It stopped at the 67th epoch with the results as displayed in the following pictures:

![](https://miro.medium.com/v2/resize:fit:1225/1*p8b8lvuXBPuGsg9zYLZfcw.png)

Total Loss Graph (Blue: Training, Red: Validation)

![](https://miro.medium.com/v2/resize:fit:1225/1*XlCzzZU9zPWhklGMoKK6Jw.png)

Binary Segmentation Loss Graph (Blue: Training, Red: Validation)

![](https://miro.medium.com/v2/resize:fit:1225/1*bd0atZE1vmYBHOvfq0_HJQ.png)

Instance Segmentation Loss Graph (Blue: Training, Red: Validation)

![](https://miro.medium.com/v2/resize:fit:1225/1*RAL5Aev0R8FQH_qatvp6Dg.png)

Mean Loss Graph (Blue: Training, Red: Validation)

![](https://miro.medium.com/v2/resize:fit:1225/1*ESquHWE3UbmCpd-HhN9AeQ.png)

Instance Segmentation Accuracy Graph (Blue: Training, Red: Validation)

The training results are as follows:

- Training loss: 2.114
- Training binary segmentation loss: 0.00294
- Training instance segmentation loss: 2.085
- Training mean IoU: 0.527
- Training instance segmentation accuracy: 0.9628
- Validation loss: 2.134
- Validation binary segmentation loss: 0.00320
- Validation instance segmentation loss: 2.101
- Validation mean IoU: 0.5261
- Validation instance segmentation accuracy: 0.9621.

With the mean IoU for binary segmentation at around 50% and accuracy for instance segmentation at 96%, it is enough for me to say that the model was able to learn to predict lanes on images.

# **TESTING**

Here are some of the lane prediction results from the model that I trained using LaneNet:

![](https://miro.medium.com/v2/resize:fit:1225/1*bEqOi8XFwAp9oCP0xfX9OQ.png)

Results of The Lane Prediction using LaneNet from Test Dataset

The video of the lane prediction implementation in the testing dataset: [https://youtu.be/wjdV2fNFGuY](https://youtu.be/wjdV2fNFGuY)

The resulting output was quite accurate in predicting the lanes on the images. In addition to that, the processing time was quite fast, which clocked in at 15.38 fps on average.

# **REFERENCES**

1. Comparison of UNet, ENet, and BoxENet for Segmentation of Mast Cells in Scans of Histological Slices ([https://arxiv.org/abs/1909.06840](https://arxiv.org/abs/1909.06840))
2. Towards End-to-End Lane Detection: an Instance Segmentation Approach ([https://arxiv.org/abs/1802.05591](https://arxiv.org/abs/1802.05591))
3. Semantic Instance Segmentation with a Discriminative Loss Function ([https://arxiv.org/abs/1708.02551](https://arxiv.org/abs/1708.02551))
4. Simple UNet Model by Dr. Sreenivas Bhattiprolu ([https://github.com/bnsreenu/python_for_microscopists/blob/master/204-207simple_unet_model.py](https://github.com/bnsreenu/python_for_microscopists/blob/master/204-207simple_unet_model.py))
5. CARLA Lane Detection Dataset Generation by Markus Heck ([https://github.com/Glutamat42/Carla-Lane-Detection-Dataset-Generation](https://github.com/Glutamat42/Carla-Lane-Detection-Dataset-Generation))
6. LaneNet-Lane-Detection by MaybeShewill-CV ([https://github.com/MaybeShewill-CV/lanenet-lane-detection](https://github.com/MaybeShewill-CV/lanenet-lane-detection))
7. LaneNet in PyTorch By AndreasKlintberg ([https://github.com/klintan/pytorch-lanenet](https://github.com/klintan/pytorch-lanenet))