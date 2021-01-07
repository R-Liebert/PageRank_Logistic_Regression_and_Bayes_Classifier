import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


# Functions used

def gamma_dist(x, alpha, beta):
    gamma = (1 / ((beta ** alpha) * math.factorial(alpha - 1))) \
            * (x ** (alpha - 1)) * np.exp(-x / beta)
    return gamma


def norm_dist(x, my, sigma2):
    norm = (1 / (np.sqrt(2 * np.pi) * np.sqrt(sigma2))) * np.exp((-(x - my) ** 2) / (2 * sigma2))

    return norm


# import training set and create dataframes for ones and zeroes

opt_df = pd.read_csv('optdigits-1d-train.csv', sep=' ', names=['label', 'x'], dtype=float)
ones_df = opt_df[opt_df.label == 1].sort_values('x', ascending=True)
zeroes_df = opt_df[opt_df.label == 0].sort_values('x', ascending=True)

# import test set

test_df = pd.read_csv('optdigits-1d-test.csv', names=['x'], dtype=float)

# count numbers of 1 and 0 in training set for calculating prior probabilities

n0 = np.sum(opt_df['label'] == 0)
n1 = np.sum(opt_df['label'] == 1)

# given parameters in task

alpha = 9

# calculate parameter estimations

beta_hat = 1 / (n0 * alpha) * opt_df[opt_df['label'] == 0].sum()['x']

my_hat = 1 / n1 * opt_df[opt_df['label'] == 1].sum()['x']

sigma2_hat = 1 / n1 * np.sum((ones_df['x'] - my_hat) ** 2)

# Oppg 2b
# calculate prior probabilities

pp_0 = n0 / (n0 + n1)
pp_1 = n1 / (n0 + n1)
print('Oppg 2b)\n Utregnet Beta hat: {0}\n Utregnet My_hat {1}\n'
      ' Utregnet sigma^2_hat: {2}\n Prior-probabilities P(C0): {3}\n'
      ' Prior-probabilities P(C1): {4}'.format(beta_hat, my_hat, sigma2_hat, pp_0, pp_1))

# Oppg 2c
# Plot the values in a histogram and in there respective distributions
gamma_dist_zeroes = gamma_dist(zeroes_df['x'], alpha, beta_hat)
norm_dist_ones = norm_dist(ones_df['x'], my_hat, sigma2_hat)

plt.plot(zeroes_df['x'], gamma_dist_zeroes)
plt.hist(zeroes_df['x'], 50, density=True)
plt.plot(ones_df['x'], norm_dist_ones)
plt.hist(ones_df['x'], 50, density=True)
plt.title('Histogram for 0 [Orange] and 1 [Red]')
plt.legend(['Gamma dist.', 'Normal dist.'])

plt.show()

# Oppg 2d
# Make a Bayes classifier
# create a vector of the training-set values

x_train = opt_df['x'].to_numpy(dtype=float)

# check if training-set values belongs to class 0 according to the Bayes classifier

pred_value = pp_0 * gamma_dist(x_train, alpha, beta_hat) - pp_1 * norm_dist(x_train, my_hat, sigma2_hat)

# create a vector containing predicted labels

pred_value_label = np.zeros((len(x_train), 1), dtype=int)

for i in range(pred_value.shape[0]):
    if pred_value[i] >= 0:
        pred_value_label[i][0] = 0
    else:
        pred_value_label[i][0] = 1

# add predicted labels to dataframe

opt_df.loc[:, 'Pred_label'] = pred_value_label.astype(int)

# make confusion matrix

confusion_matrix = pd.crosstab(opt_df['label'], opt_df['Pred_label'], rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix)

# calculate precision, recall and accuracy

precision = (confusion_matrix[0.0][0]/(confusion_matrix[0.0][0]+confusion_matrix[0.0][1]))
recall = (confusion_matrix[0.0][0]/(confusion_matrix[0.0][0]+confusion_matrix[1.0][0]))
accuracy = (confusion_matrix[0.0][0]+confusion_matrix[1.0][1])/len(opt_df['Pred_label'])

print('Oppg 2c)\n Precision: {0}\n Recall: {1}\n Accuracy: {2}\n F1-score: {3}'.format(precision, recall, accuracy, F1))

# Oppg 2d
# Create a vector of the test-set values

x_test = test_df['x'].to_numpy(dtype=float)

# check if test-set values belongs to class 0 according to the Bayes classifier

pred_test_value = pp_0 * gamma_dist(x_test, alpha, beta_hat) - pp_1 * norm_dist(x_test, my_hat, sigma2_hat)

# create a vector containing predicted labels

pred_test_value_label = np.zeros((len(x_test), 1), dtype=int)

for i in range(pred_test_value.shape[0]):
    if pred_test_value[i] >= 0:
        pred_test_value_label[i][0] = 0
    else:
        pred_test_value_label[i][0] = 1

# add predicted labels to dataframe and to a .csv file for reading the secret message
test_label_df = pd.DataFrame(pred_test_value_label, columns=['label'])
test_label_df['label'].to_csv("pred_label.csv")
