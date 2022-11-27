import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
import numpy as np

# 0 - nothing, 1 -scissors, 2 - rock, 3 - paper

m_games = 500
m_game = 5
memory_nn = []
memory_my = []

for i in range(m_games):
    memory_my.append([])
    memory_nn.append([])
    for j in range(m_game):
        memory_my[len(memory_my)-1].append(0)
        memory_nn[len(memory_nn)-1].append(0)


class SRP_NN(nn.Module):
    def __init__(self):
        super(SRP_NN, self).__init__()

        self.l1 = nn.Linear(in_features=6*m_game*m_games, out_features=1000)
        self.l2 = nn.Linear(in_features=1000, out_features=700)
        self.l = nn.Linear(in_features=700, out_features=3)
        self.relu = nn.ReLU()

    def forward(self, x):
        new_x = []
        for i in range(len(x)):
            if x[i] == 0:
                new_x.append(0)
                new_x.append(0)
                new_x.append(0)
            if x[i] == 1:
                new_x.append(1)
                new_x.append(0)
                new_x.append(0)
            if x[i] == 2:
                new_x.append(0)
                new_x.append(1)
                new_x.append(0)
            if x[i] == 3:
                new_x.append(0)
                new_x.append(0)
                new_x.append(1)
        new_x = torch.tensor(new_x, dtype=torch.float32)

        new_x = self.relu(self.l1(new_x))
        new_x = self.relu(self.l2(new_x))
        new_x = self.l(new_x)

        return new_x


def newline():
    for a in range(len(memory_nn) - 1):
        for b in range(len(memory_nn[0])):
            memory_my[len(memory_my) - a - 1][b] = memory_my[len(memory_my) - a - 2][b]
            memory_nn[len(memory_nn) - a - 1][b] = memory_nn[len(memory_nn) - a - 2][b]
    for b in range(len(memory_nn[0])):
        memory_my[0][b] = 0
        memory_nn[0][b] = 0


my_win = 0
my_lose = 0
path = "Artyyr.pth"
net = torch.load(path)
op = torch.optim.SGD(params=net.parameters(), lr=0.1)

i = True
while i:
    net = torch.load(path)
    op = torch.optim.SGD(params=net.parameters(), lr=0.1)
    print("1 - Scissors\n2 - Rock\n3 - Paper\n")
    j = True
    while j:
        x1 = list(chain.from_iterable(memory_my))
        x2 = list(chain.from_iterable(memory_nn))
        x = list(chain.from_iterable([x1, x2]))
        choice_nn = net(x)
        #print(net(x))
        choice = int(input("You: "))

        if choice_nn[0] > choice_nn[1] and choice_nn[0] > choice_nn[2]:
            print("NN: Scissors")
            choice_num = 1
        elif choice_nn[1] > choice_nn[0] and choice_nn[1] > choice_nn[2]:
            print("NN: Rock")
            choice_num = 2
        elif choice_nn[2] > choice_nn[0] and choice_nn[2] > choice_nn[1]:
            print("NN: Paper")
            choice_num = 3
        else:
            print("NN: Scissors")
            choice_num = 1
        if choice == 1 and choice_num == 1:
            memory_my[0][4] = memory_my[0][3]
            memory_my[0][3] = memory_my[0][2]
            memory_my[0][2] = memory_my[0][1]
            memory_my[0][1] = memory_my[0][0]
            memory_my[0][0] = 1

            memory_nn[0][4] = memory_nn[0][3]
            memory_nn[0][3] = memory_nn[0][2]
            memory_nn[0][2] = memory_nn[0][1]
            memory_nn[0][1] = memory_nn[0][0]
            memory_nn[0][0] = 1

            op.zero_grad()
            loss = F.cross_entropy(net(x), torch.tensor([0., 1., 0.], dtype=torch.float32))
            loss.backward()
            op.step()
        elif choice == 2 and choice_num == 2:
            memory_my[0][4] = memory_my[0][3]
            memory_my[0][3] = memory_my[0][2]
            memory_my[0][2] = memory_my[0][1]
            memory_my[0][1] = memory_my[0][0]
            memory_my[0][0] = 2

            memory_nn[0][4] = memory_nn[0][3]
            memory_nn[0][3] = memory_nn[0][2]
            memory_nn[0][2] = memory_nn[0][1]
            memory_nn[0][1] = memory_nn[0][0]
            memory_nn[0][0] = 2

            op.zero_grad()
            loss = F.cross_entropy(net(x), torch.tensor([0., 0., 1.], dtype=torch.float32))
            loss.backward()
            op.step()
        elif choice == 3 and choice_num == 3:
            memory_my[0][4] = memory_my[0][3]
            memory_my[0][3] = memory_my[0][2]
            memory_my[0][2] = memory_my[0][1]
            memory_my[0][1] = memory_my[0][0]
            memory_my[0][0] = 3

            memory_nn[0][4] = memory_nn[0][3]
            memory_nn[0][3] = memory_nn[0][2]
            memory_nn[0][2] = memory_nn[0][1]
            memory_nn[0][1] = memory_nn[0][0]
            memory_nn[0][0] = 3

            op.zero_grad()
            loss = F.cross_entropy(net(x), torch.tensor([1., 0., 0.], dtype=torch.float32))
            loss.backward()
            op.step()
        elif choice == 1 and choice_num == 2:
            memory_my[0][4] = memory_my[0][3]
            memory_my[0][3] = memory_my[0][2]
            memory_my[0][2] = memory_my[0][1]
            memory_my[0][1] = memory_my[0][0]
            memory_my[0][0] = 1

            memory_nn[0][4] = memory_nn[0][3]
            memory_nn[0][3] = memory_nn[0][2]
            memory_nn[0][2] = memory_nn[0][1]
            memory_nn[0][1] = memory_nn[0][0]
            memory_nn[0][0] = 2

            op.zero_grad()
            loss = F.cross_entropy(net(x), torch.tensor([0., 1., 0.], dtype=torch.float32))
            loss.backward()
            op.step()

            newline()
            print("#########\nLose")
            my_lose = my_lose + 1
            break
        elif choice == 1 and choice_num == 3:
            memory_my[0][4] = memory_my[0][3]
            memory_my[0][3] = memory_my[0][2]
            memory_my[0][2] = memory_my[0][1]
            memory_my[0][1] = memory_my[0][0]
            memory_my[0][0] = 1

            memory_nn[0][4] = memory_nn[0][3]
            memory_nn[0][3] = memory_nn[0][2]
            memory_nn[0][2] = memory_nn[0][1]
            memory_nn[0][1] = memory_nn[0][0]
            memory_nn[0][0] = 3

            op.zero_grad()
            loss = F.cross_entropy(net(x), torch.tensor([0., 1., 0.], dtype=torch.float32))
            loss.backward()
            op.step()

            newline()
            print("#########\nWin")
            my_win = my_win + 1
            break
        elif choice == 2 and choice_num == 1:
            memory_my[0][4] = memory_my[0][3]
            memory_my[0][3] = memory_my[0][2]
            memory_my[0][2] = memory_my[0][1]
            memory_my[0][1] = memory_my[0][0]
            memory_my[0][0] = 2

            memory_nn[0][4] = memory_nn[0][3]
            memory_nn[0][3] = memory_nn[0][2]
            memory_nn[0][2] = memory_nn[0][1]
            memory_nn[0][1] = memory_nn[0][0]
            memory_nn[0][0] = 1

            op.zero_grad()
            loss = F.cross_entropy(net(x), torch.tensor([0., 0., 1.], dtype=torch.float32))
            loss.backward()
            op.step()

            newline()
            print("#########\nWin")
            my_win = my_win + 1
            break
        elif choice == 2 and choice_num == 3:
            memory_my[0][4] = memory_my[0][3]
            memory_my[0][3] = memory_my[0][2]
            memory_my[0][2] = memory_my[0][1]
            memory_my[0][1] = memory_my[0][0]
            memory_my[0][0] = 2

            memory_nn[0][4] = memory_nn[0][3]
            memory_nn[0][3] = memory_nn[0][2]
            memory_nn[0][2] = memory_nn[0][1]
            memory_nn[0][1] = memory_nn[0][0]
            memory_nn[0][0] = 3

            op.zero_grad()
            loss = F.cross_entropy(net(x), torch.tensor([0., 0., 1.], dtype=torch.float32))
            loss.backward()
            op.step()

            newline()
            print("#########\nLose")
            my_lose = my_lose + 1
            break
        elif choice == 3 and choice_num == 2:
            memory_my[0][4] = memory_my[0][3]
            memory_my[0][3] = memory_my[0][2]
            memory_my[0][2] = memory_my[0][1]
            memory_my[0][1] = memory_my[0][0]
            memory_my[0][0] = 3

            memory_nn[0][4] = memory_nn[0][3]
            memory_nn[0][3] = memory_nn[0][2]
            memory_nn[0][2] = memory_nn[0][1]
            memory_nn[0][1] = memory_nn[0][0]
            memory_nn[0][0] = 2

            op.zero_grad()
            loss = F.cross_entropy(net(x), torch.tensor([1., 0., 0.], dtype=torch.float32))
            loss.backward()
            op.step()

            newline()
            print("#########\nWin")
            my_win = my_win + 1
            break
        elif choice == 3 and choice_num == 1:
            memory_my[0][4] = memory_my[0][3]
            memory_my[0][3] = memory_my[0][2]
            memory_my[0][2] = memory_my[0][1]
            memory_my[0][1] = memory_my[0][0]
            memory_my[0][0] = 3

            memory_nn[0][4] = memory_nn[0][3]
            memory_nn[0][3] = memory_nn[0][2]
            memory_nn[0][2] = memory_nn[0][1]
            memory_nn[0][1] = memory_nn[0][0]
            memory_nn[0][0] = 1

            op.zero_grad()
            loss = F.cross_entropy(net(x), torch.tensor([1., 0., 0.], dtype=torch.float32))
            loss.backward()
            op.step()

            newline()
            print("#########\nLose")
            my_lose = my_lose + 1
            break
        #print(memory_my)
        #print(memory_nn)
    print("Win: " + str(my_win) + " | Lose: " + str(my_lose))
    if my_lose == 0:
        print("Win: " + str(100))
    else:
        print("Win: " + str(round((my_win/(my_lose + my_win))*100)) + "%")
    torch.save(net, path)
    #print(memory_my)
    #print(memory_nn)