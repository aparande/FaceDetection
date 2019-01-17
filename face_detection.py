import numpy as np
import pickle
from viola_jones import ViolaJones

def train(t):
    with open("training.pkl", 'rb') as f:
        training = pickle.load(f)
    clf = ViolaJones(T=t)
    clf.train(training, 2429, 4548)

    clf.save(str(t))
    correct = 0
    for x, y in training:
        correct += 1 if clf.classify(x) == y else 0
    print("Classified %d out of %d training examples" % (correct, len(training)))

def test(filename):
    with open("test.pkl", 'rb') as f:
        test = pickle.load(f)
    
    clf = ViolaJones.load(filename)
    correct = 0
    for x, y in training:
        correct += 1 if clf.classify(x) == y else 0
    print("Classified %d out of %d training examples" % (correct, len(training)))
