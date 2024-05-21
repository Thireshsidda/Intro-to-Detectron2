# Intro to Detectron2
The developers of Detectron2 are Meta’s Facebook AI Research (FAIR) team, who have stated that “Our goal with Detectron2 is to support the wide range of cutting-edge object detection and segmentation models available today, but also to serve the ever-shifting landscape of cutting-edge research.” 

Detectron2 is a deep learning model built on the Pytorch framework, which is said to be one of the most promising modular object detection libraries being pioneered. Meta has suggested that Detectron2 was created to help with the research needs of Facebook AI under the aegis of FAIR teams – that said, it has been widely adopted in the developer community, and is available to public teams.
![detectron2-for-object-detection-and-other-computer-vision-tasks](https://github.com/Thireshsidda/Intro-to-Detectron2/assets/92287626/1473b7b6-e10c-447d-9425-b7ba0849e109)
##### Meta’s FAIR team developed Detectron2 as a state-of-the-art object detection framework.


# Detectron2: What’s Inside? 
## Structure
Detectron2 is written in Pytorch, whereas the initial Detectron model was built on Caffe2. Developer teams have noted that Caffe2 and Pytorch have now merged, making that difference sort of moot. Additionally, Detectron2 offers several backbone options, including ResNet, ResNeXt, and MobileNet.

##### The three main structures to point out in the Detectron2 architecture are as follows: 
**Backbone Network:** The Detectron2 backbone network extracts feature maps at different scales from the input image. 

**Regional Proposal Network (RPN):** object regions are detected from multi-scale features. 

**ROI Heads:** Region of Interest heads process feature maps generated from selected regions of the image. This is done by extracting and reshaping feature maps based on proposal boxes into a variety of fixed-size features, refining box positions, and classification outcomes through fully connected layers.

As the image is downsampled by successive CNN layers, the features stay stable, and task-specific heads help to generate outputs. R-CNNs use items like bounding boxes to delineate parts of an image and help with object detection. Detectron2 supports what’s called two-stage detection, and is good at using training data to build model capabilities for this kind of computer vision. As a practical source of image processing capabilities, Detectron2 pioneers the practice of collecting image data sets. It then teaches the model to “look” at them and “see” things.


# Key Features 
*Modular Design:** Detectron2’s modular architecture allows users to easily experiment with model architectures, loss functions, and training techniques. 

**High Performance:** Detectron2 achieves state-of-the-art performance on various benchmarks, which include COCO dataset (Common Objects in Context), LVIS (Large Vocabulary Instance Segmentation) – with over 1.5 million object instances and 250,000 examples of key point data. 

**Support for Custom Datasets:** the Detectron2 framework provides tools for working with custom datasets. 

**Pre-trained Models:** Detectron2’s model zoo comes with a collection of pre-trained models for each computer vision task supported (see the full list for each computer vision task below). 

**Efficient Inference:** Detectron2 includes optimizations for efficient inference, meaning that it performs well for deployment in production environments with real-time or low-latency requirements. 

**Active Development Community:** Due to its open-source nature, Detectron2 has an active development community with contributions from users around the world.

![Object-Detection-Results-on-MS-COCO-Dataset-768x493](https://github.com/Thireshsidda/Intro-to-Detectron2/assets/92287626/d6019197-ab53-408c-aa4a-fe2bfab08f7a)

# Detectron2 Computer Vision Tasks The Detectron2 model can perform several computer vision tasks including: 
**Object Detection:** Identifying and localizing objects with bounding boxes. 

**Semantic Segmentation: assigning each pixel in an image a class label for precise delineation and understanding of a scene’s objects and regions. 

**Instance Segmentation:** Identifying and delineating individual objects within an image, assigning a label to each instance, and outlining boundaries. 

**Panoptic Segmentation:** combining semantic and instance segmentation, providing a more in-depth analysis of the scene by labeling each object instance and background regions by class. 

**Keypoint Detection:** identifying and localizing points or features of interest within an image. 

**DensePose Estimation:*8 assigning dense relations between points on the surface of objects and pixels in an image for a detailed understanding of object geometry and texture.

### Below is a list of pre-trained models for each computer vision task provided in the Detectron2 model zoo.

![Screenshot (99)](https://github.com/Thireshsidda/Intro-to-Detectron2/assets/92287626/cebdea5a-cb2b-4170-8184-1073dcd9209e)

# Starting with Detectron2
### Step 1: Install Dependencies
```
Install PyTorch: using a pip package with specific versions depending on system and hardware requirements. 
Install Torchvision: using a pip package as well.
```

### Step 2: Install Detectron2
Clone the Detectron2 GitHub repository: use Git to clone the Detectron2 repository from GitHub to receive access to source code and latest updates. Clone the Detectron2 GitHub repo using the following command:
```
git clone https://github.com/facebookresearch/detectron2.git
```
Or, install the pip package: installing the pip package will ensure that you’re using the latest version.

See the official [installation guide](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).


### Step 3: Set Up Dataset Prepare dataset: 
organize the dataset into the required structure and format compatible with Detectron2’s data-loading utilities. Convert data to Detectron2 format

### Step 4: Create Config File 
Customize config file: define model architecture (Faster R-CNN, Mask R-CNN), specify hyperparameters (e.g., learning rate, batch size), and provide paths to dataset and other necessary resources. 

Fine-tune configurations: based on your requirements.

### Step 5: Training or Inference 
Train model: use configured settings and dataset to train a new model from scratch. 

Perform inference: generate predictions, such as object detections or image segmentations, depending on the task.

### Step 6: Evaluation and Fine-tuning 
Evaluate model performance to inform fine-tuning and adjusting of hyperparameters as needed.

### Step 7: Deployment 
Deploy Detectron2 in your application or system and ensure that the deployment environment meets hardware and software requirements for running the models efficiently.

![yolo-object-detection-background](https://github.com/Thireshsidda/Intro-to-Detectron2/assets/92287626/9fe30ef3-8825-467f-a3af-a26f0b9c3483)

# Here are some hands-on experiments using Detectron2
## Install using Docker

Another great way to install Detectron2 is by using Docker. Docker is great because you don't need to install anything locally, which allows you to keep your machine nice and clean.

If you want to run Detectron2 with Docker you can find a Dockerfile and docker-compose.yml file in the [docker directory of the repository](https://github.com/facebookresearch/detectron2/tree/master/docker).

For those of you who also want to use Jupyter notebooks inside their container, I created a custom Docker configuration, which automatically starts Jupyter after running the container. If you're interested you can find the files in the [docker directory](https://github.com/TannerGilbert/Object-Detection-and-Image-Segmentation-with-Detectron2/tree/master/docker).

## Inference with pre-trained model

* [Detectron2 Inference with pretrained model](Detectron2_inference_with_pre_trained_model.ipynb).
* [Detect from Webcam or Video](detect_from_webcam_or_video.py)
* [Detectron2 RestAPI](deploy/rest-api)

## Training on a custom dataset

* [Detectron2 train on a custom dataset](Detectron2_train_on_a_custom_dataset.ipynb).
* [Detectron2 train with data augmentation](Detectron2_Train_on_a_custom_dataset_with_data_augmentation.ipynb)
* [Detectron2 Chess Detection](Detectron2_Detect_Chess_Detection.ipynb)
* [Detectron2 Vehicle Detection](Detectron2_Vehicle_Detection.ipynb)

## D2Go

* [D2GO_Introduction](D2Go/D2GO_Introduction.ipynb)
* [D2Go_Train_Microcontroller_Detector](D2Go/D2Go_Train_Microcontroller_Detector.ipynb)


# Detectron 2 Real-World Applications

Practically speaking, the real-world applications of Detectron2 are nearly endless. 
* Autonomous Driving: In self-driving car systems or semi-supervised models, Detectron2 can help to identify pedestrians, road signs, roadways, and vehicles with precision. Tying object detection and recognition to other elements of AI provides a lot of assistance to human drivers. It does not matter whether fully autonomous driving applications are “ready for prime time.”
* Robotics: A robot’s capabilities are only as good as its computer vision. The ability of robots to move pieces or materials, navigate complex environments, or achieve processing or packaging tasks has to do with what the computer can take individually from image data. With that in mind, Detectron2 is a core part of many robotics frameworks for AI cognition that make these robots such effective helpers.
* Security: As you can imagine, Detectron2 is also critically helpful in security imaging. A major part of AI security systems involves monitoring and filtering through image data. The better AI can do this, the more capable it is in identifying threats and suspicious activity.
* Safety Fields: In safety fields, Detectron2 results can help to prevent emergencies, for example, in forest fire monitoring, or flood research. Then, generally, there are a lot of maritime industry applications, and biology research applications, too. Object detection technology helps researchers to observe natural systems and measure goals and objectives.
* Education: Detectron2 can be a powerful example of applying AI in education to teach its capabilities. As students start to understand how AI entities “see” and start to understand the world around them, it prepares them to live in a world where we increasingly coexist and interact with intelligent machines.
  

![computer-vision-surveillance-security-applications](https://github.com/Thireshsidda/Intro-to-Detectron2/assets/92287626/2a932e02-db62-44f2-a72e-a37aaa289664)
##### Figure: Video surveillance and object tracking for security applications with Detectron2

## Author

👤 **Thiresh Sidda**

* LinkedIn: [@ThireshSidda](https://www.linkedin.com/in/thiresh-sidda)
* GitHub: [@ThireshSidda](https://github.com/Thireshsidda)
