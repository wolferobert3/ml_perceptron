import math
import numpy as np
import csv
import os
import matplotlib.pyplot as plt

def get_accuracy(weight_vector, data, labels):
    correct = 0
    for i in range(len(data)):
        prediction = np.dot(weight_vector, data[i])
        classification = 1 if (1/(1 + math.exp(prediction * -1))) >= 0.5 else 0
        correct += max(0, int(classification == labels[i]))
    return (correct)

def learn_weights(weight_vector, learning_rate, training_data, labels):
    update_vector = np.array(weight_vector)
    for i in range(len(training_data)):
        prediction = np.dot(weight_vector, training_data[i])
        classification = 1 if (1/(1 + math.exp(prediction * -1))) >= 0.5 else 0
        update_size = learning_rate * (labels[i] - classification)
        change_vector = np.multiply(training_data[i], update_size)
        update_vector = np.add(update_vector, change_vector)
    return update_vector

def k_fold(learning_rate, data, labels, epochs):

    mean_weight_vector = np.zeros(len(data[0]))
    
    for k in range(0, 10):
        weight_vector = np.array([0.5 for i in range(len(data[0]))])
        testing_data = np.array([data[i] for i in range(k, len(data), 10)])
        testing_labels = np.array([labels[i] for i in range(k, len(data), 10)])
        training_data = np.array([data[i] for i in range(len(data)) if i % 10 != k])
        training_labels = [labels[i] for i in range(len(data)) if i % 10 != k]
        for _ in range(0, epochs):
            weight_vector = learn_weights(weight_vector, learning_rate, training_data, training_labels)
        validation = get_accuracy(weight_vector, testing_data, testing_labels)
        mean_weight_vector = np.add(mean_weight_vector, weight_vector)

    mean_weight_vector = np.divide(mean_weight_vector, 10)

    return mean_weight_vector

def confidence(weight_vector, data):
    confidence_list = []
    classification_list = []
    for i in range(len(data)):
        prediction = np.dot(weight_vector, data[i])
        sigmoid_prediction = (1/(1 + math.exp(prediction * -1)))
        confidence = np.abs(0.5 - sigmoid_prediction) * 2
        classification = 1 if sigmoid_prediction >= 0.5 else 0
        confidence_list.append(confidence)
        classification_list.append(classification)
    conf_v_class = [confidence_list, classification_list]
    return conf_v_class

dir_path = os.path.dirname(os.path.realpath(__file__))
with open(dir_path + '\\iris.data', 'r', newline='') as iris_file:
    iris_data = list(csv.reader(iris_file))

iris_data.pop(-1)
iris_arr = np.array(iris_data[0:50] + iris_data[100:])
word_labels = list(iris_arr[:, -1])
num_labels = [0 if word_labels[i] == 'Iris-setosa' else 1 for i in range(len(word_labels))]
iris_arr[:, -1] = 1.0
iris_arr = iris_arr.astype(float)

conf_weights = k_fold(0.00005, iris_arr, num_labels, 100)
confidence_array = confidence(conf_weights, iris_arr)
plt.scatter(confidence_array[0], confidence_array[1], alpha=0.3)
plt.xlabel("Confidence")
plt.ylabel("Classification")
plt.title("Confidence vs. Classification")
plt.yticks(ticks=[0,1])
plt.show()
print(confidence_array)

lr_rates = [.005, .001, .00005]
accuracy_list = []

for i in range(len(lr_rates)):
    accuracy_history = []
    for j in range(0, 100):
        epoch_weights = k_fold(lr_rates[i], iris_arr, num_labels, j)
        accuracy_history.append(get_accuracy(epoch_weights, iris_arr, num_labels))
    accuracy_list.append(accuracy_history)

accuracy_table = np.array(accuracy_list)
print(accuracy_table)

final_weights = k_fold(.00005, iris_arr, num_labels, 100)
accuracy = get_accuracy(final_weights, iris_arr, num_labels)

linsep_data = np.array(iris_arr[:, 0:4])
linsep_labels = ['r' if i == 0 else 'k' for i in num_labels]
plt.scatter(linsep_data[:, 0], linsep_data[:, 1], s = 20 * linsep_data[:, 2], c = linsep_data[:, 3], cmap="plasma", edgecolors=linsep_labels)
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Linear Separability of Iris Flower Classes")
plt.show()

l1 = np.array(accuracy_table[0])
l2 = np.array(accuracy_table[1])
l3 = np.array(accuracy_table[2])

plt.plot(l1, label = "0.005")
plt.plot(l2, label = "0.001")
plt.plot(l3, label = "0.00005")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Classification Accuracy by Learning Rate: 100 Epochs")
plt.xticks(ticks=[0, 20, 40, 60, 80, 100])
plt.legend()
plt.show()