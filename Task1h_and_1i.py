import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Functions

def plot_regression_line(x, y, b):
    # plotting the actual points as scatter plot
    plt.scatter(x, y, color="m", marker="o")

    # predicted response vector
    y_pred = b[0] + b[1]*(x)

    # plotting the regression line
    plt.plot(x, y_pred, color="g")

    # putting labels
    plt.xlabel('ln(Pr)')
    plt.ylabel('Elo')

    # function to show plot
    plt.show()


# Import and clean csv
df = pd.read_csv('chess-games-elo.csv', sep=',', names=['id', 'elo'])
PR_df = pd.read_csv('PageRank.csv', sep=',')

df.loc[:, 'PR_Score'] = PR_df['PR_Score'].astype(float)
df['ln_PR'] = np.log(df['PR_Score'])

# Observations
x = df['ln_PR']
X = np.c_[np.ones((df.shape[0], 1)), df['ln_PR']]
y = df['elo']
beta = np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, y))

# estimating coefficients

y_pred = beta[0] + x*beta[1]
print("Estimated coefficients:\nb_0 = {}  \nb_1 = {}".format(beta[0], beta[1]))


plot_regression_line(x, y, beta)

# Calculate Mean Square Error and Rsquared
MSE = np.sum((y_pred - y)**2) / x.shape[0]
e = np.sum((y-y_pred)**2) / np.sum((y-np.mean(y))**2)
r2 = 1-e

print('MSE =', MSE)
print("Calculated r2-score: ", r2)
