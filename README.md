# FaceDetection
Face Detection in Python using the Viola-Jones algorithm on the CBCL Face Database published by MIT's Center for Biological and Computational Learning. Learn how it works by reading https://medium.com/@parandea1.7/understanding-and-implementing-the-viola-jones-image-classification-algorithm-85621f7fe20b

# Contents
- viola_jones.py
  - An implementation of the Viola-Jones algorithm
  - Works in Python2 and Python3
  - Viola, Paul, and Michael Jones. "Rapid object detection using a boosted cascade of simple features." Computer Vision and Pattern Recognition, 2001. CVPR 2001. Proceedings of the 2001 IEEE Computer Society Conference on. Vol. 1. IEEE, 2001. Works
- face_detection.py
  - Methods to train and test a ViolaJones classifier on the training and test datasets

# Data
The data is described at http://cbcl.mit.edu/software-datasets/FaceData2.html, and I downloaded from www.ai.mit.edu/courses/6.899/lectures/faces.tar.gz and compiled into pickle files.

Each image is 19x19 and greyscale. There are Training set:  2,429 faces, 4,548 non-faces
Test set: 472 faces, 23,573 non-faces 

- training.pkl
  - An array of tuples. The first element of each tuple is a numpy array representing the image. The second element is its clasification (1 for face, 0 for non-face)
  - 2429 face images, 4548 non-face images

- test.pkl
  - An array of tuples. The first element of each tuple is a numpy array representing the image. The second element is its clasification (1 for face, 0 for non-face)
  - 472 faces, 23573 non-face images

# Results
The hyperparameter T for the ViolaJones class represents how many weak classifiers it uses. 

For T=10, the model achieved 85.5% accuracy on the training set and 78% accuracy on the test set.

For T=50, the model achieved 93% accuracy on both the training and test set.

