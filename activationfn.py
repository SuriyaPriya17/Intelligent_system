
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
 return 1/(1+np.exp(-x))

def tanh(x):
 return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
def relu(x):
  return np.maximum(0, x)
ip = -1
op = sigmoid(ip)
print(op)
x = np.linspace(-10, 10, 1000) 

y = sigmoid(x)

plt.plot(x, y)

plt.xlabel("x")
plt.ylabel("sigmoid(x)")
plt.title("Sigmoid Function Graph")

plt.grid(True)

plt.show()

y = tanh(x)
plt.plot(x, y)

plt.xlabel("x")
plt.ylabel("tanh(x)")
plt.title("Tanh Function Graph")

plt.grid(True)

plt.show()

y = relu(x)

plt.plot(x, y)

plt.xlabel("x")
plt.ylabel("relu(x)")
plt.title("ReLU Function Graph")

plt.grid(True)


plt.show()