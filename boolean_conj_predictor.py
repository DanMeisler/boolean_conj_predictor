import os
import numpy as np


def predictor_variable_tostring(variable_value_pair):
    if variable_value_pair[1] == 1:
        return "x%s" % variable_value_pair[0]
    elif variable_value_pair[1] == 2:
        return "not(x%s)" % variable_value_pair[0]
    else:
        return ""


def output_conjunction_predictor(conjunction_predictor):
    conjunction_predictor_string = ",".join(filter(None, map(predictor_variable_tostring, enumerate(conjunction_predictor))))
    with open(r"./output.txt", "w") as output_file:
        output_file.write(conjunction_predictor_string)


if __name__ == "__main__":
    file_path = os.path.join(r"./trainingData/example1.txt")
    training_examples = np.loadtxt(file_path)
    d = training_examples.shape[1] - 1
    x_of_training_data = training_examples[:, :d].T
    y_of_training_data = training_examples[:, -1].reshape(training_examples.shape[0])
    conjunction_predictor = np.zeros(d)
    output_conjunction_predictor(conjunction_predictor)
    print x_of_training_data
    print
    print y_of_training_data
