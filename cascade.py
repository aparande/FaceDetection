from viola_jones import ViolaJones
import pickle
"""
A Python implementation of the "Attentional Cascade" mentioned in 
Viola, Paul, and Michael Jones. "Rapid object detection using a boosted cascade of simple features." Computer Vision and Pattern Recognition, 2001. CVPR 2001. Proceedings of the 2001 IEEE Computer Society Conference on. Vol. 1. IEEE, 2001.
Works in both Python2 and Python3
"""

class CascadeClassifier():
    def __init__(self, layers):
        self.layers = layers
        self.clfs = []

    def train(self, training):
        pos, neg = [], []
        for ex in training:
            if ex[1] == 1:
                pos.append(ex)
            else:
                neg.append(ex)
        
        for feature_num in self.layers:
            if len(neg) == 0:
                print("Stopping early. FPR = 0")
                break
            clf = ViolaJones(T=feature_num)
            clf.train(pos+neg, len(pos), len(neg))
            self.clfs.append(clf)
            false_positives = []
            for ex in neg:
                if self.classify(ex[0]) == 1:
                    false_positives.append(ex)
            neg = false_positives

    def classify(self, image):
        for clf in self.clfs:
            if clf.classify(image) == 0:
                return 0
        return 1

    def save(self, filename):
        """
        Saves the classifier to a pickle
          Args:
            filename: The name of the file (no file extension necessary)
        """
        with open(filename+".pkl", 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        A static method which loads the classifier from a pickle
          Args:
            filename: The name of the file (no file extension necessary)
        """
        with open(filename+".pkl", 'rb') as f:
            return pickle.load(f)