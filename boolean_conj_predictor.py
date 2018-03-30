import os
import numpy as np


def output_conjunction_predictor(conjunction_predictor):
    conjunction_predictor_string = ",".join(
        map(lambda index: "x%d" % (index / 2 + 1) if index % 2 == 0 else "not(x%d)" % (index / 2 + 1),
            np.where(conjunction_predictor == 1)[0]))
    with open(r"./output.txt", "w") as output_file:
        output_file.write(conjunction_predictor_string)


def calculate_label_from_instance(conjunction_predictor, instance):
    instance_conjunction = np.array(map(lambda x: [1, 0] if x else [0, 1], instance)) \
        .reshape(conjunction_predictor.shape)
    return 1 if np.array_equal(conjunction_predictor, conjunction_predictor & instance_conjunction) else 0


def consistency_algorithm(d, x_of_training_data, y_of_training_data):
    conjunction_predictor = np.ones(d * 2).astype(int)
    for instance_index, instance in enumerate(x_of_training_data):
        if (y_of_training_data[instance_index] == 1) and \
                (calculate_label_from_instance(conjunction_predictor, instance) == 0):
            for i, x in enumerate(instance):
                if x == 1:
                    conjunction_predictor[i * 2 + 1] = 0
                elif x == 0:
                    conjunction_predictor[i * 2] = 0
    output_conjunction_predictor(conjunction_predictor)


if __name__ == "__main__":
    file_path = os.path.join(r"./trainingData/example1.txt")
    training_examples = np.loadtxt(file_path).astype(int)
    d = training_examples.shape[1] - 1
    x_of_training_data = training_examples[:, :-1]
    y_of_training_data = training_examples[:, -1].reshape(training_examples.shape[0])
    consistency_algorithm(d, x_of_training_data, y_of_training_data)
