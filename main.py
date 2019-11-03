import torch
import torchvision
from torchvision import transforms, datasets


train = datasets.MNIST("", train=True, download=True,
                       transform = transforms.Compose([transforms.ToTensor()]))

test = datasets.MNIST("", train=False, download=True,
                       transform = transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

  def __init__(self):
    super().__init__()
    self.inputLayer = nn.Linear(28*28, 64)
    self.hiddenLayer1 = nn.Linear(64, 64)
    self.hiddenLayer2 = nn.Linear(64, 64)
    self.outputLayer = nn.Linear(64, 10)

  def forward(self, x):
    x = F.relu(self.inputLayer(x))
    x = F.relu(self.hiddenLayer1(x))
    x = F.relu(self.hiddenLayer2(x))
    x = self.outputLayer(x)

    return F.log_softmax(x, dim=1)

net = Net()
print(net)

X = torch.rand((28,28))
X = X.view(-1,28*28)
output = net(X)

output

import torch.optim as optim

optimizer = optim.Adam(net.parameters(), lr=0.001)

EPOCHS = 3 #aantal stappen door een dataset

for epoch in range(EPOCHS):
  for data in trainset:
    # data is a batch of featuresets and labels
    X, y = data
    net.zero_grad()
    output = net(X.view(-1, 28*28))
    loss = F.nll_loss(output, y)
    loss.backward() #back propogation
    optimizer.step() #adjust weights for us
  print(loss)

correct = 0
total = 0

with torch.no_grad():
  for data in trainset:
    X, y = data
    output = net(X.view(-1, 784))
    for idx, i in enumerate(output):
      if torch.argmax(i) == y[idx]:
        correct+= 1
      total += 1
      
print("Accuracy: ", round(correct/total, 3))

import matplotlib.pyplot as plt
plt.imshow(X[9].view(28,28))
plt.show()

print(torch.argmax(net(X[9].view(-1,784))[0]))

for a in trainset:
  print(a)
  break

x, y = a[0][0], a[1][0]

print(y)

import matplotlib.pyplot as plt

plt.imshow(a[0][0].view(28,28))
plt.show()
print(x[0][0].shape)

total = 0
counter_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
for data in trainset:
  xS, ys = data
  for y in ys:
    counter_dict[int(y)] +=1
    total+=1
print(counter_dict)

for i in counter_dict:
  print(f"{i}: {counter_dict[i]/total*100}")
