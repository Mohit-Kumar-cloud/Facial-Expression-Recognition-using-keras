# Facial-Expression-Recognition-using-keras

This Project is Created using TensorFlow, Keras and open-cv to perform live video Facial Expression Detection

Project is Fully functional By using pre-trained models created Using FER-2013 Dataset, However a utility 'training.py' is also providided in order to generate Models from Any other datasets



# Project Structure
.  
|__ Training-Dataset  
|__ Validation-Dataset  
|__ training.py  
|__ detect.py  
|__ FER.py  
|__ haarcascade_frontalface_default.xml  
|__ Emotion_little_vgg.h5  

Emotion_little_vgg.h5 can also be generated by  training.py

# Getting Started
To get a local copy up and running follow these simple example steps.
Clone the repository
```
git clone https://github.com/Mohit-Kumar-cloud/Facial-Expression-Recognition-using-keras.git
```

Create a virtual environment
```
python3 -m virtualenv venv
```
Install the requirements
```
pip install -r requirements.txt
```
To run on a particular image
```
python detect.py 'path_to_img'
```
To run a live Video Detection
```
python FER.py
```
