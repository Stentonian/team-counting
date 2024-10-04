"""
We have N people, and teams of size 1,2,3,...,t
We want the probability of k teams occurring.

Example:
for N=10 and only teams of size 1 & 2
1. first, put everyone into 1 team; there are 10 teams then, and only 1 possible configuration
2. next, have 1 team of 2 and the rest singles; there are 9 teams then, and nCr(10,2) possible configurations
3. ...
So we have a map of num_teams->count
"""

from itertools import combinations
import math
import matplotlib.pyplot as plt
from decimal import Decimal


def count_teams_size_2(N):
    """
    Here is the problem simplified to just picking teams of size 1 & 2.

    This gives us the formula of how to count the number of ways of splitting P people into teams of 2:
    https://math.stackexchange.com/questions/1234696/number-of-ways-you-can-form-pairs-with-a-group-of-people-when-certain-people-can

    We have to loop from 1 team of 2 to 5 teams of 2, and use the above formula for each iteration.
    """
    counts_size_2 = [0] * (1 + N)

    num_teams_of_size_2 = lambda n: \
        math.factorial(2*n) / (math.factorial(n) * 2**n)

    for i in range(0, N // 2 + 1):
        # num_teams_size_2 + num_teams_size_1
        num_teams = i + (N - 2 * i)
        count = math.comb(N, 2 * i) * num_teams_of_size_2(i)
        counts_size_2[num_teams] += count

    return counts_size_2


def count_teams_size_3(N):
    """
    Here is the problem simplified to just picking teams of size 1, 2 & 3.

    We have to update our counting formula:
    https://math.stackexchange.com/questions/2663826/counting-the-number-of-ways-to-partition-n-integers-to-n-3-triples

    We also change our loop to first loop over teams of 3, then of 2
    """
    counts_size_3 = [0] * (1 + N)

    num_teams_of_size_2 = lambda n: \
        math.factorial(2*n) / (math.factorial(n) * 2**n)

    num_teams_of_size_3 = lambda n: \
        math.factorial(3*n) / (math.factorial(n) * math.factorial(3)**n)

    for i in range(0, N // 3 + 1):
        for j in range(0, (N - 3 * i) // 2 + 1):
            # num_teams_size_3 + num_teams_size_2 + num_teams_size_1
            num_teams = i + j + (N - 3 * i - 2 * j)

            count = math.comb(N, 3 * i) * num_teams_of_size_3(i)
            count *= math.comb(N - 3 * i, 2 * j) * num_teams_of_size_2(j)

            counts_size_3[num_teams] += count

    return counts_size_3


def num_occurances(n, k):
    """
    This is the general formula for any team size k and number of people k*n

    ways = (k*n)! / (n! ((k*n)!)^n)

    We wrap the values in Decimal because they get large quickly, and then we get errors like:
    > OverflowError: integer division result too large for a float
    """
    return Decimal(math.factorial(k*n)) / \
        (Decimal(math.factorial(n)) * Decimal(math.factorial(k))**n)


# total number of people
N = 200

counts = [0] * (1 + N)


# call like this: count_teams(max_team_size, N, N, 1)
def count_teams(team_size, people_left, num_teams, count):
    """
    Recursive function for counting for any number of teams, assuming teams are of size 1,2,3,...,max_team_size
    """
    if team_size == 1:
        counts[num_teams] += count
        return

    for i in range(0, people_left // team_size + 1):
        next_num_teams = num_teams - i * (team_size - 1)

        next_count = \
            count \
            * math.comb(people_left, team_size * i) \
            * num_occurances(i, team_size)

        count_teams(team_size - 1, \
                    people_left - team_size * i, \
                    next_num_teams, \
                    next_count)


count_teams(4, N, N, 1)
total = sum(counts)
probabilities = list(map(lambda x: x / total, counts))

for i, p in enumerate(probabilities):
    if p > 0.005:
        print(f"num teams: {i}; prob: {p}")

num_teams = range(0, N + 1)

# Create the graph
plt.figure(figsize=(10, 6))
plt.plot(num_teams, probabilities)
plt.xlabel("Number of Teams (x)")
plt.ylabel("Probability f(x)")
plt.title("Probability of Forming x Teams")
plt.grid(True)
plt.show()

# =======================

# Now we try to find the line of best fit for the function that takes number of people as input, and gives the most probable number of teams.

exit()

from sklearn.linear_model import LinearRegression
import numpy as np

x = []
y = []
max_team_size = 4
for N in range(50, 300, 5):
    counts = [0] * (1 + N)
    count_teams(max_team_size, N, N, 1)
    i = counts.index(max(counts))
    x.append(N)
    y.append(i)

x = np.array(x).reshape(-1, 1)  # Reshape for scikit-learn
y = np.array(y)
model = LinearRegression()
model.fit(x, y)

# Get the slope and intercept
slope = model.coef_[0]
intercept = model.intercept_

# Print results
print(f"Slope: {slope}")
print(f"Y-intercept: {intercept}")

# Plot data and best fit line
plt.scatter(x, y, label="Data Points")
plt.plot(x, model.predict(x), color='red', label="Best Fit Line")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.title("Line of Best Fit")
plt.show()
