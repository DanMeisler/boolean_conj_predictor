import os
import numpy as np

file_path = os.path.join(r"./trainingData/example1.txt")
training_examples = np.loadtxt(file_path)
print training_examples
