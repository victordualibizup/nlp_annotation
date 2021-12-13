def calculate_accuracy(*, real_values, predicted_values):
    real_vector = real_values.copy()
    counter = 0
    vector_size = len(real_vector)

    for i, classes in enumerate(real_vector):
        if classes == predicted_values[i]:
            counter += 1

    accuracy = round(counter / vector_size, 2)

    return accuracy
