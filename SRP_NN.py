import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
import numpy as np

# 0 - nothing, 1 - scissors, 2 - rock, 3 - paper
m_games = 10
m_game = 5
torch.autograd.set_detect_anomaly(True)


class Game():
    def __init__(self, hidden_size, load_path=""):
        super(Game, self).__init__()

        self.path = load_path
        self.memory_nn = []
        self.memory_my = []

        try:
            self.model = torch.load(load_path)
            print("Loaded from " + load_path)
        except FileNotFoundError:
            self.model = SRP_NN(hidden_size=hidden_size)
            print("Created new nn with a saving path " + load_path)
        self.op = torch.optim.SGD(params=self.model.parameters(), lr=0.01)

        self.nn_win = 0
        self.my_win = 0
        self.story = []  # 0 - draw, 1 - nn win, 2 - my win

        for i in range(m_games):
            self.memory_my.append([])
            self.memory_nn.append([])
            for j in range(m_game):
                self.memory_my[-1].append(0)
                self.memory_nn[-1].append(0)

    def get_memory(self):
        x1 = list(chain.from_iterable(self.memory_my))
        x2 = list(chain.from_iterable(self.memory_nn))

        return np.array(list(chain.from_iterable([x1, x2])))

    def update(self, choice):
        x = self.get_memory()
        res = self.model(x)
        choice_num = np.argmax(res.detach().numpy(), axis=0) + 1

        if choice == 1 and choice_num == 1:
            self.memory_my[0][4] = self.memory_my[0][3]
            self.memory_my[0][3] = self.memory_my[0][2]
            self.memory_my[0][2] = self.memory_my[0][1]
            self.memory_my[0][1] = self.memory_my[0][0]
            self.memory_my[0][0] = 1

            self.memory_nn[0][4] = self.memory_nn[0][3]
            self.memory_nn[0][3] = self.memory_nn[0][2]
            self.memory_nn[0][2] = self.memory_nn[0][1]
            self.memory_nn[0][1] = self.memory_nn[0][0]
            self.memory_nn[0][0] = 1

            self.op.zero_grad()
            loss = F.cross_entropy(res.view((1, 3)), torch.tensor([1], dtype=torch.int64))
            loss.backward()
            self.op.step()
            self.story.append(0)
        elif choice == 2 and choice_num == 2:
            self.memory_my[0][4] = self.memory_my[0][3]
            self.memory_my[0][3] = self.memory_my[0][2]
            self.memory_my[0][2] = self.memory_my[0][1]
            self.memory_my[0][1] = self.memory_my[0][0]
            self.memory_my[0][0] = 2

            self.memory_nn[0][4] = self.memory_nn[0][3]
            self.memory_nn[0][3] = self.memory_nn[0][2]
            self.memory_nn[0][2] = self.memory_nn[0][1]
            self.memory_nn[0][1] = self.memory_nn[0][0]
            self.memory_nn[0][0] = 2

            self.op.zero_grad()
            loss = F.cross_entropy(res.view((1, 3)), torch.tensor([2], dtype=torch.int64))
            loss.backward()
            self.op.step()
            self.story.append(0)
        elif choice == 3 and choice_num == 3:
            self.memory_my[0][4] = self.memory_my[0][3]
            self.memory_my[0][3] = self.memory_my[0][2]
            self.memory_my[0][2] = self.memory_my[0][1]
            self.memory_my[0][1] = self.memory_my[0][0]
            self.memory_my[0][0] = 3

            self.memory_nn[0][4] = self.memory_nn[0][3]
            self.memory_nn[0][3] = self.memory_nn[0][2]
            self.memory_nn[0][2] = self.memory_nn[0][1]
            self.memory_nn[0][1] = self.memory_nn[0][0]
            self.memory_nn[0][0] = 3

            self.op.zero_grad()
            loss = F.cross_entropy(res.view((1, 3)), torch.tensor([0], dtype=torch.int64))
            loss.backward()
            self.op.step()
            self.story.append(0)
        elif choice == 1 and choice_num == 2:
            self.memory_my[0][4] = self.memory_my[0][3]
            self.memory_my[0][3] = self.memory_my[0][2]
            self.memory_my[0][2] = self.memory_my[0][1]
            self.memory_my[0][1] = self.memory_my[0][0]
            self.memory_my[0][0] = 1

            self.memory_nn[0][4] = self.memory_nn[0][3]
            self.memory_nn[0][3] = self.memory_nn[0][2]
            self.memory_nn[0][2] = self.memory_nn[0][1]
            self.memory_nn[0][1] = self.memory_nn[0][0]
            self.memory_nn[0][0] = 2

            self.op.zero_grad()
            loss = F.cross_entropy(res.view((1, 3)), torch.tensor([1], dtype=torch.int64))
            loss.backward()
            self.op.step()
            self.nn_win += 1
            self.story.append(1)

            newline(self.memory_my, self.memory_nn)
        elif choice == 1 and choice_num == 3:
            self.memory_my[0][4] = self.memory_my[0][3]
            self.memory_my[0][3] = self.memory_my[0][2]
            self.memory_my[0][2] = self.memory_my[0][1]
            self.memory_my[0][1] = self.memory_my[0][0]
            self.memory_my[0][0] = 1

            self.memory_nn[0][4] = self.memory_nn[0][3]
            self.memory_nn[0][3] = self.memory_nn[0][2]
            self.memory_nn[0][2] = self.memory_nn[0][1]
            self.memory_nn[0][1] = self.memory_nn[0][0]
            self.memory_nn[0][0] = 3
            self.my_win += 1
            self.story.append(2)

            self.op.zero_grad()
            loss = F.cross_entropy(res.view((1, 3)), torch.tensor([1], dtype=torch.int64))
            loss.backward()
            self.op.step()

            newline(self.memory_my, self.memory_nn)
        elif choice == 2 and choice_num == 1:
            self.memory_my[0][4] = self.memory_my[0][3]
            self.memory_my[0][3] = self.memory_my[0][2]
            self.memory_my[0][2] = self.memory_my[0][1]
            self.memory_my[0][1] = self.memory_my[0][0]
            self.memory_my[0][0] = 2

            self.memory_nn[0][4] = self.memory_nn[0][3]
            self.memory_nn[0][3] = self.memory_nn[0][2]
            self.memory_nn[0][2] = self.memory_nn[0][1]
            self.memory_nn[0][1] = self.memory_nn[0][0]
            self.memory_nn[0][0] = 1

            self.op.zero_grad()
            loss = F.cross_entropy(res.view((1, 3)), torch.tensor([2], dtype=torch.int64))
            loss.backward()
            self.op.step()
            self.my_win += 1
            self.story.append(2)

            newline(self.memory_my, self.memory_nn)
        elif choice == 2 and choice_num == 3:
            self.memory_my[0][4] = self.memory_my[0][3]
            self.memory_my[0][3] = self.memory_my[0][2]
            self.memory_my[0][2] = self.memory_my[0][1]
            self.memory_my[0][1] = self.memory_my[0][0]
            self.memory_my[0][0] = 2

            self.memory_nn[0][4] = self.memory_nn[0][3]
            self.memory_nn[0][3] = self.memory_nn[0][2]
            self.memory_nn[0][2] = self.memory_nn[0][1]
            self.memory_nn[0][1] = self.memory_nn[0][0]
            self.memory_nn[0][0] = 3

            self.op.zero_grad()
            loss = F.cross_entropy(res.view((1, 3)), torch.tensor([2], dtype=torch.int64))
            loss.backward()
            self.nn_win += 1
            self.story.append(1)

            self.op.step()

            newline(self.memory_my, self.memory_nn)
        elif choice == 3 and choice_num == 2:
            self.memory_my[0][4] = self.memory_my[0][3]
            self.memory_my[0][3] = self.memory_my[0][2]
            self.memory_my[0][2] = self.memory_my[0][1]
            self.memory_my[0][1] = self.memory_my[0][0]
            self.memory_my[0][0] = 3

            self.memory_nn[0][4] = self.memory_nn[0][3]
            self.memory_nn[0][3] = self.memory_nn[0][2]
            self.memory_nn[0][2] = self.memory_nn[0][1]
            self.memory_nn[0][1] = self.memory_nn[0][0]
            self.memory_nn[0][0] = 2

            self.op.zero_grad()
            loss = F.cross_entropy(res.view((1, 3)), torch.tensor([0], dtype=torch.int64))
            loss.backward()
            self.op.step()
            self.my_win += 1
            self.story.append(2)

            newline(self.memory_my, self.memory_nn)
        elif choice == 3 and choice_num == 1:
            self.memory_my[0][4] = self.memory_my[0][3]
            self.memory_my[0][3] = self.memory_my[0][2]
            self.memory_my[0][2] = self.memory_my[0][1]
            self.memory_my[0][1] = self.memory_my[0][0]
            self.memory_my[0][0] = 3

            self.memory_nn[0][4] = self.memory_nn[0][3]
            self.memory_nn[0][3] = self.memory_nn[0][2]
            self.memory_nn[0][2] = self.memory_nn[0][1]
            self.memory_nn[0][1] = self.memory_nn[0][0]
            self.memory_nn[0][0] = 1

            self.op.zero_grad()
            loss = F.cross_entropy(res.view((1, 3)), torch.tensor([0], dtype=torch.int64))
            loss.backward()
            self.op.step()
            self.nn_win += 1
            self.story.append(1)

            newline(self.memory_my, self.memory_nn)

        try:
            torch.save(self.model, self.path)
        except FileNotFoundError:
            print("nn is not saved")
        print("srp nn saved in " + self.path)

        return choice_num


class SRP_NN(nn.Module):
    def __init__(self, hidden_size):
        super(SRP_NN, self).__init__()

        self.l1 = nn.Linear(in_features=8*m_game*m_games, out_features=hidden_size)
        self.l2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.l = nn.Linear(in_features=hidden_size, out_features=3)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = np.array(x)

        res = np.zeros((x.size, 4), dtype=int)
        res[np.arange(x.size), x] = 1
        res = res.flatten()

        new_x = torch.tensor(res, dtype=torch.float32)

        new_x = self.relu(self.l1(new_x))
        new_x = self.dropout(self.relu(self.l2(new_x)))
        new_x = self.l(new_x)

        new_x = self.softmax(new_x)

        return new_x


def newline(memory_my, memory_nn):
    for a in range(len(memory_nn) - 1):
        for b in range(len(memory_nn[0])):
            memory_my[len(memory_my) - a - 1][b] = memory_my[len(memory_my) - a - 2][b]
            memory_nn[len(memory_nn) - a - 1][b] = memory_nn[len(memory_nn) - a - 2][b]
    for b in range(len(memory_nn[0])):
        memory_my[0][b] = 0
        memory_nn[0][b] = 0
