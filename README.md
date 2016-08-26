# faze
A python module to analyze faces, provides functionalities like: face detection, landmark detection, pose estimation etc.

## Requirements
* OpenCV ( 3.0.0+ )
* scikit-learn ( scikit-learn>=0.15.1)
* Dlib ( 18.16+ )
* You should also download the shape\_predictor\_68\_face\_landmarks.dat from [here](http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2).
* Numpy ( numpy>=1.8.0 )

## Structure
* **detect**: Face/landmark detection, face pose estimation, frontal transform of faces.
* **recog**: A submodule which caters to all the recognition methods. Currently, the following are supported.
    * emotion

## Getting started
Clone this repository to get started.
```
$ git clone https://github.com/Aravind-Suresh/faze
$ cd faze
```
To start using the functionalities, import and use. For example,
```
# imports necessary modules and methods
import faze

::

# dets stores the ROIs of faces detected
dets = faze.detect.faces(img)

# ROI is the first face detected
roi = dets[0]

# landmarks holds the facial landmarks of the face ROI
landmarks = faze.detect.landmarks(img, roi, "/path/to/shape_predictor")
```

## How to contribute
If you think of any improvements to the project, feel free to make a **pull request**.
