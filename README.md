# Design of Eye Blink Detection using Computer Vision and Transfer Learning
**Long working hours** and **inadequate supervision** contribute to an increased risk of incidents and accidents due to fatigue. Thus, a **real-time fatigue measurement instrument** is needed to monitor and assess the workers' levels of fatigue during work. This is a project that was submitted for a bachelor's degree in Industrial Engineering at Institut Teknologi Bandung.

## Demo


https://user-images.githubusercontent.com/48656293/135448479-7054aeb3-ff33-4e4f-b502-1b21fa0480f8.mp4


<!-- 
## Proposed Algorithm for Eye Blink Detection
![proposed-algorithm-english](https://user-images.githubusercontent.com/48656293/135438500-20649408-8ad6-4a61-8d9b-ea635cce549b.jpg)

## Training the Classifier
### Preparing the training data

**The Closed Eyes in the Wild (CEW) Dataset** (Song, et.al., 2014) contains 2423 subjects face that are collected directly from the Internet, consists of 1192 subjects with both eyes are closed and 1231 subjects with both eyes are opened.
![face_sample](https://user-images.githubusercontent.com/48656293/135443417-e31bc4c8-fddb-4a1b-9470-8138bfd32b07.png)

### Extracting eye features
Eye features are extracted using **HOG-based object detection algorithm** (Dalal & Triggs, 2005) with the help of **68 facial landmark points** (focusing only on the eye features which are **36-47**).

![Untitled](https://user-images.githubusercontent.com/48656293/135444578-85e7c7a1-3b9e-406d-9d9d-be68c752b56f.png)

### Training process
The extracted eye features data is split randomly into training data (80%) and validation data (20%). **MobileNet**, a CNN-based machine learning classifier trained on ImageNet classification problem, is trained as the eye status classifier. Resulting trained model achieves **96.7%** accuracy on validation dataset. MobileNet can be imported using `tensorflow` module and this line of code: 
[Tensorflow Documentation Link](https://www.tensorflow.org/api_docs/python/tf/keras/applications/mobilenet/MobileNet)
```python
tensorflow.keras.applications.mobilenet.MobileNet()
```

## References
1. F.Song, X.Tan, X.Liu and S.Chen, Eyes Closeness Detection from Still Images with Multi-scale Histograms of Principal Oriented Gradients, Pattern Recognition, 2014.
2. Navneet Dalal, Bill Triggs. Histograms of Oriented Gradients for Human Detection. International Conference on Computer Vision & Pattern Recognition (CVPR ’05), Jun 2005, San Diego, United States. pp.886–893, [10.1109/CVPR.2005.177](https://ieeexplore.ieee.org/document/1467360). [inria-00548512](https://hal.inria.fr/inria-00548512) -->
