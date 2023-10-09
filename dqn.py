import random
import pygame
import time
import copy
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

render = True
if render:
    pygame.init()

board_init = [
    [0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0],
    [0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0],
    [0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0],
    [0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0],
    [0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0],
    [0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0],
    [0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0],
    [0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0],
    [0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0],
    [0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0],
    [0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0],
    [0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0],
    [0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0],
    [0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0],
    [0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0],
    [0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0],
    [0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0],
    [0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0],
    [0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0],
    [0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0],
    [0, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
]

pieces = [
    [
        [0, 11, 0],
        [11, 11, 11],
        [0, 0, 0]
    ],
    [
        [12, 12, 0],
        [0, 12, 12],
        [0, 0, 0]
    ],
    [
        [13, 13],
        [13, 13]
    ],
    [
        [0, 14, 14],
        [14, 14, 0],
        [0, 0, 0]
    ],
    [
        [0, 0, 0, 0],
        [15, 15, 15, 15],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ],
    [
        [16, 16, 0],
        [0, 16, 0],
        [0, 16, 0]
    ],
    [
        [0, 17, 17],
        [0, 17, 0],
        [0, 17, 0]
    ]
]

# pieces = [
#     [
#         [13, 13],
#         [13, 13]
#     ]
# ]

w = 800
h = 600

if render:
    surface = pygame.display.set_mode((w, h))

def draw_board():
    s = 20 # for sidelength of each tile
    g = 2 # for gaps between tiles
    hs = w//2 - s*len(board[0])//2 # for horizontal shift of board on surface
    vs = h//2 - s*len(board)//2 # for vertical shift of board on surface
    for i in range(len(board)):
        for j in range(len(board[0])):
            pygame.draw.rect(surface, (0, 0, 0), pygame.Rect(hs+s*j, vs+s*i, s, s))
            if board[i][j] == 10:
                pygame.draw.rect(surface, (155, 155, 155), pygame.Rect(hs+s*j, vs+s*i, s-g, s-g))
            if board[i][j] == 11:
                pygame.draw.rect(surface, (255, 0, 255), pygame.Rect(hs+s*j, vs+s*i, s-g, s-g))
            if board[i][j] == 12:
                pygame.draw.rect(surface, (255, 0, 0), pygame.Rect(hs+s*j, vs+s*i, s-g, s-g))
            if board[i][j] == 13:
                pygame.draw.rect(surface, (255, 255, 0), pygame.Rect(hs+s*j, vs+s*i, s-g, s-g))
            if board[i][j] == 14:
                pygame.draw.rect(surface, (0, 255, 0), pygame.Rect(hs+s*j, vs+s*i, s-g, s-g))
            if board[i][j] == 15:
                pygame.draw.rect(surface, (0, 155, 255), pygame.Rect(hs+s*j, vs+s*i, s-g, s-g))
            if board[i][j] == 16:
                pygame.draw.rect(surface, (255, 155, 0), pygame.Rect(hs+s*j, vs+s*i, s-g, s-g))
            if board[i][j] == 17:
                pygame.draw.rect(surface, (0, 0, 255), pygame.Rect(hs+s*j, vs+s*i, s-g, s-g))
            if not board[i][j] in [10, 11, 12, 13, 14, 15, 16, 17, 0]:
                pygame.draw.rect(surface, (255, 255, 255), pygame.Rect(hs+s*j, vs+s*i, s-g, s-g))
    pygame.display.flip()

def piece_generator():
    while True:
        random.shuffle(pieces)
        for piece in pieces:
            yield piece

def collision():
    for i in range(len(piece)):
        for j in range(len(piece[0])):
            if board[x+i][y+j] >= 20:
                return True
    return False

def place_piece():
    for i in range(len(piece)):
        for j in range(len(piece[0])):
            board[x+i][y+j] += piece[i][j]

def remove_piece():
    for i in range(len(piece)):
        for j in range(len(piece[0])):
            board[x+i][y+j] -= piece[i][j]

def clear_line():
    global reward
    for i in range(len(board)-4, -1, -1):
        line_saturated = True
        for j in range(len(board[0])-4, 2, -1):
            if board[i][j] == 0:
                line_saturated = False
        if line_saturated:
            print('line clear')
            reward += 1.0
            for k in range(i, -1, -1):
                for l in range(len(board[0])-4, 2, -1):
                    board[k][l] = board[k-1][l]

def move_left():
    global piece, x, y
    remove_piece()
    y -= 1
    place_piece()
    if collision():
        remove_piece()
        y += 1
        place_piece()

def move_right():
    global piece, x, y
    remove_piece()
    y += 1
    place_piece()
    if collision():
        remove_piece()
        y -= 1
        place_piece()

def move_rotate():
    global piece, x, y
    remove_piece()
    piece = list(zip(*piece[::-1]))
    place_piece()
    if collision():
        remove_piece()
        for _ in range(3):
            piece = list(zip(*piece[::-1]))
        place_piece()

def move_down():
    global piece, x, y
    remove_piece()
    x += 1
    place_piece()
    if collision():
        remove_piece()
        x -= 1
        place_piece()

def move_drop():
    global piece, x, y
    while not collision():
        remove_piece()
        x += 1
        place_piece()
    remove_piece()
    x -= 1
    place_piece()

def process_input():
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                move_left()
            if event.key == pygame.K_RIGHT:
                move_right()
            if event.key == pygame.K_UP:
                move_rotate()
            if event.key == pygame.K_DOWN:
                move_down()
            if event.key == pygame.K_SPACE:
                move_drop()
            if event.key == pygame.K_q:
                pygame.quit()
                sys.exit()

def gravity():
    global piece, x, y, generator, board, t0
    if time.time() - t0 > 1:
        t0 = time.time()
        remove_piece()
        x += 1
        place_piece()
        if collision():
            remove_piece()
            x -= 1
            if x <= x_init:
                print('game over')
                terminal = True
                board = copy.deepcopy(board_init)
                generator = piece_generator()
                piece, x, y = next(generator), x_init, y_init
                place_piece()
                return 
            place_piece()
            piece, x, y = next(generator), x_init, y_init 
            place_piece()

def dqn_gravity():
    global piece, x, y, generator, board, t0, elapsed_times
    t0 = time.time()
    remove_piece()
    x += 1
    place_piece()
    if collision():
        remove_piece()
        x -= 1
        if x <= x_init:
            elapsed_time = time.time()-t0
            elapsed_times.append(elapsed_time)
            print('game over:', elapsed_time)
            plt.plot(elapsed_times)
            plt.savefig('game_times.png')
            t0 = 0
            terminal = True
            board = copy.deepcopy(board_init)
            generator = piece_generator()
            piece, x, y = next(generator), x_init, y_init
            place_piece()
            return 
        place_piece()
        piece, x, y = next(generator), x_init, y_init 
        place_piece()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear((20+3)*(10+3+3), (20+3)*(10+3+3))
        self.fc2 = nn.Linear((20+3)*(10+3+3), (20+3)*(10+3+3))
        self.fc3 = nn.Linear((20+3)*(10+3+3), (20+3)*(10+3+3))
        self.fc4 = nn.Linear((20+3)*(10+3+3), 5)
        self.dropout1 = nn.Dropout(0.9)
        self.dropout2 = nn.Dropout(0.9)
        self.dropout3 = nn.Dropout(0.9)
        torch.nn.init.eye_(self.fc1.weight)
        torch.nn.init.eye_(self.fc2.weight)
        torch.nn.init.eye_(self.fc3.weight)
        # torch.nn.init.eye_(self.fc4.weight)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.silu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.silu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = F.silu(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        output = F.log_softmax(x, dim=1)
        return output

def process_dqn_input():
    global board, model, reward, optimizer, terminal
    pred = model(torch.tensor(board, dtype=torch.float32).unsqueeze(0).to(device))[0]
    move = torch.argmax(pred).item()
    options = {0: move_left, 1: move_right, 2: move_rotate, 3: move_down, 4: move_drop}
    options[move]()
    y = copy.deepcopy(pred.detach())
    if terminal:
        y[move] = reward
    else:
        gamma = 0.95
        y[move] = reward + gamma * torch.argmax(model(torch.tensor(board, dtype=torch.float32).unsqueeze(0).to(device))).item()
    terminal = False
    reward = 0.0
    optimizer.zero_grad()
    loss = F.mse_loss(pred, y)
    loss.backward()
    optimizer.step()

device = 'cuda:0'
model = Net().to(device)
generator = piece_generator()
board = copy.deepcopy(board_init)
piece = next(generator)
x_init = 0
y_init = len(board[0])//2 - len(piece)//2
x = x_init
y = y_init
place_piece()
t0 = time.time()
reward = 0.0
terminal = False
optimizer = optim.AdamW(model.parameters(), lr=1.0)
elapsed_times = []

while True:
    if render:
        process_input()
    process_dqn_input()
    # gravity()
    dqn_gravity()
    clear_line()
    if render:
        draw_board()
