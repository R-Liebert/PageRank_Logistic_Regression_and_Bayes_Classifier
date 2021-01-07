import numpy as np
import pandas as pd

# Functions used

def power_iter(A, num_it):
    # Create a pi_vec with random values
    pi_vec = np.random.rand(A.shape[1])
    B = np.matrix(np.transpose(A))**(num_it+1)

    pi_vec_1 = np.matmul(B, pi_vec)

    return pi_vec_1

def google_matrix(S):
    """ Create the Google Matrix to solve 0 < p < 1 """
    alpha = 0.85  # Set alpha
    E = np.ones(S.shape)  # Make a matrix of ones in the shape of S
    E = E / len(E)  # Make the rows/columns sum to 1

    G = alpha * S + (1 - alpha) * E

    return G


# Import files
Chess_df = pd.read_csv("chess-games.csv", skiprows=16, sep=' ', names=['ID_White', 'ID_Black', 'Result'])
Chess = Chess_df.to_numpy(dtype=float)
PlayerName = pd.read_csv('chess-games-names.csv', sep=',', names=['ID', 'Name'])

# Maximum ID of player and +1 due to id value 0
N = (Chess_df['ID_White'].max() if Chess_df['ID_White'].max() > Chess_df['ID_Black'].max() else Chess_df[
    'ID_Black'].max())+1

# Length of games set
L = np.shape(Chess)

# Prepare H of Zeroes
A = np.zeros((N, N))

# Make link matrix
for i in range(L[0]):
    if Chess[i][2] == 0:
        A[int(Chess[i][0]), int(Chess[i][1])] += 1
    elif Chess[i][2] == 1:
        A[int(Chess[i][1]), int(Chess[i][0])] += 1

# Task 1f
# Run power method over link matrix and print top ten
task_1f = (power_iter(A, 50)).T

# Sort and name
Af = pd.DataFrame(data=task_1f)
Af.columns = ['PR_Score']
Af.loc[:, 'ID'] = Af.index
Af.loc[:, 'Name'] = PlayerName['Name'].astype(str)
Af_sorted = Af.sort_values('PR_Score', ascending=False)
Af_sorted.loc[:, 'Rank'] = Af.index+1

print('Topp ti ranking ved bruk av link matrise uten å fjerne dangling nodes eller summering av rader til 1')
print(Af_sorted.head(10))
print('\n')


# I make A into S so i can make it stochastic and irreducible and keeping A
S = A
sum_row = S.sum(axis=1, dtype=int)

# Sum each row to one and remove dangling nodes
for i in range(N):
    if sum_row[i] == 0:
        for k in range(N):
            S[i][k] = 1 / N
        continue
    for j in range(N):
        if S[i][j] > 0:
            S[i][j] = S[i][j] / (sum_row[i])

# Run through the google algoritmen
G = google_matrix(S)

#
task_1g = (power_iter(G, 50)).T

# Sort and name
Ag = pd.DataFrame(data=task_1g)
Ag.columns = ['PR_Score']
Ag.loc[:, 'ID'] = Ag.index
Ag.loc[:, 'Name'] = PlayerName['Name'].astype(str)
Ag.to_csv('PageRank.csv')
Ag_sorted = Ag.sort_values('PR_Score', ascending=False)
Ag_sorted.loc[:, 'Rank'] = Ag.index+1

print('Topp ti ranking ved bruk av Google-matrisen')
print(Ag_sorted.head(10))
