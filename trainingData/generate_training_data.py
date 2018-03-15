import os
import numpy as np


def generate_random_boolean_conjunction(d):
    return np.random.choice(3, d)


def calculate_label_from_instance(conjunction_predictor, instance):
    conjunction_predictor_copy = np.copy(conjunction_predictor)
    for i in range(len(conjunction_predictor_copy)):
        if conjunction_predictor_copy[i] == 2:
            conjunction_predictor_copy[i] = instance[i]
    return 1 if np.array_equal(conjunction_predictor_copy, instance) else 0


def generate_training_data_from_conjunction_predictor(conjunction_predictor, data_size):
    matrix = np.array([])
    for _ in range(data_size):
        instance = np.random.choice(2, len(conjunction_predictor))
        matrix = np.append(matrix, np.append(instance, calculate_label_from_instance(conjunction_predictor, instance)))

    return matrix.reshape((data_size, len(conjunction_predictor) + 1))


def save_training_data(training_data):
    i = 0
    while os.path.exists("example%s.txt" % i):
        i += 1
    np.savetxt("example%s.txt" % i, training_data, "%d")


if __name__ == "__main__":
    d = 3
    data_size = 15
    random_conjunction_predictor = generate_random_boolean_conjunction(d)
    print random_conjunction_predictor
    save_training_data(generate_training_data_from_conjunction_predictor(random_conjunction_predictor, data_size))
