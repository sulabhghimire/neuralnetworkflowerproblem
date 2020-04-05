from matplotlib import pyplot as plt
import numpy as np

#each point is length, width and type i.e (1 for Red and 0 for Blue)
data = [[3   ,1.5 ,1],
        [2   ,1   ,0],
        [4   ,1.5 ,1],
        [3   ,1   ,0],
        [3.5 ,.5  ,1],
        [2   ,.5  ,0],
        [5.5 ,1   ,1],
        [1   ,1   ,0]]

flower = [2.5,1]

def sigmoid(x):
    return 1/(1+np.exp(-x))
def der_sigmoid(x):
    return sigmoid(x) * (1-sigmoid(x))

#scatter data
plt.grid()
for i in range(len(data)):
    point = data[i]
    color = 'blue'
    if (point[2] == 1):
        color = 'red'
    plt.scatter(point[0], point[1], c=color)
plt.show()
# training loop
learning_rate = 0.5
costs = []

w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn()

for i in range(500000):
    ri = np.random.randint(len(data))
    point = data[ri]

    z = point[0] * w1 + point[1] * w2 + b
    pred = sigmoid(z)
    target = point[2]

    cost = (pred - point[2]) ** 2

    costs.append(cost)

    dcost_dpred = 2 * (pred - point[2])

    dpred_dz = der_sigmoid(z)

    dz_dw1 = point[0]
    dz_dw2 = point[1]
    dz_db = 1

    dcost_dw1 = dcost_dpred * dpred_dz * dz_dw1
    dcost_dw2 = dcost_dpred * dpred_dz * dz_dw2
    dcost_db = dcost_dpred * dpred_dz * dz_db

    w1 = w1 - learning_rate * dcost_dw1
    w2 = w2 - learning_rate * dcost_dw2
    b = b - learning_rate * dcost_db

    if i % 100 == 0:
        cost_sum = 0
        for j in range(len(data)):
            point = data[j]

            z = w1 * point[0] + w2 * point[1] + b
            pred = sigmoid(z)
            target = point[2]

            cost_sum += np.square(pred - target)

        costs.append(cost_sum / len(data))

for k in range(len(data)):
    point = data[k]

    z = w1 * point[0] + w2 * point[1] + b
    pred = sigmoid(z)

    print(data[k])
    print(pred)


plt.plot(costs)

result = sigmoid(w1 * flower[0] + w2 * flower[1] + b)
if result >0.5:
    print("Flower is RED")
else:
    print("Flower is BLUE")

plt.show()