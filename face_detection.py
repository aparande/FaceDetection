import numpy as np
import pickle
from viola_jones import ViolaJones
from cascade import CascadeClassifier
import time

def train_viola(t):
    with open("training.pkl", 'rb') as f:
        training = pickle.load(f)
    clf = ViolaJones(T=t)
    clf.train(training, 2429, 4548)
    evaluate(clf, training)
    clf.save(str(t))

def test_viola(filename):
    with open("test.pkl", 'rb') as f:
        test = pickle.load(f)
    
    clf = ViolaJones.load(filename)
    evaluate(clf, test)

def train_cascade(layers, filename="Cascade"):
    with open("training.pkl", 'rb') as f:
        training = pickle.load(f)
    
    clf = CascadeClassifier(layers)
    clf.train(training)
    evaluate(clf, training)
    clf.save(filename)

def test_cascade(filename="Cascade"):
    with open("test.pkl", "rb") as f:
        test = pickle.load(f)
    
    clf = CascadeClassifier.load(filename)
    evaluate(clf, test)

def evaluate(clf, data):
    correct = 0
    all_negatives, all_positives = 0, 0
    true_negatives, false_negatives = 0, 0
    true_positives, false_positives = 0, 0
    classification_time = 0

    for x, y in data:
        if y == 1:
            all_positives += 1
        else:
            all_negatives += 1

        start = time.time()
        prediction = clf.classify(x)
        classification_time += time.time() - start
        if prediction == 1 and y == 0:
            false_positives += 1
        if prediction == 0 and y == 1:
            false_negatives += 1
        
        correct += 1 if prediction == y else 0
    
    print("False Positive Rate: %d/%d (%f)" % (false_positives, all_negatives, false_positives/all_negatives))
    print("False Negative Rate: %d/%d (%f)" % (false_negatives, all_positives, false_negatives/all_positives))
    print("Accuracy: %d/%d (%f)" % (correct, len(data), correct/len(data)))
    print("Average Classification Time: %f" % (classification_time / len(data)))
