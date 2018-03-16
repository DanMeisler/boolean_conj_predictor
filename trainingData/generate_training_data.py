import os
import numpy as np

ONLY_VARIABLE = 0
ONLY_NOT_VARIABLE = 1
NONE = 2


def generate_random_boolean_conjunction(d):
    random_boolean_conjunction = []
    for _ in range(d):
        choice = np.random.randint(0, 3)
        if choice == ONLY_VARIABLE:
            random_boolean_conjunction += [1, 0]
        elif choice == ONLY_NOT_VARIABLE:
            random_boolean_conjunction += [0, 1]
        elif choice == NONE:
            random_boolean_conjunction += [0, 0]
    return np.array(random_boolean_conjunction)


def calculate_label_from_instance(conjunction_predictor, instance):
    instance_conjunction = np.array(map(lambda x: [1, 0] if x else [0, 1], instance))\
        .reshape(conjunction_predictor.shape)
    return 1 if np.array_equal(conjunction_predictor, conjunction_predictor & instance_conjunction) else 0


def generate_training_data_from_conjunction_predictor(conjunction_predictor, d, data_size):
    matrix = np.array([])
    for _ in range(data_size):
        instance = np.random.choice(2, d)
        matrix = np.append(matrix, np.append(instance, calculate_label_from_instance(conjunction_predictor, instance)))

    return matrix.reshape((data_size, d + 1))


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
    save_training_data(generate_training_data_from_conjunction_predictor(random_conjunction_predictor, d, data_size))
