import random
import pygame
import time
import copy
import sys
import numpy as np

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

pieces = [
    # [
    #     [0, 11, 0],
    #     [11, 11, 11],
    #     [0, 0, 0]
    # ],
    # [
    #     [12, 12, 0],
    #     [0, 12, 12],
    #     [0, 0, 0]
    # ],
    [
        [13, 13],
        [13, 13]
    ],
    # [
    #     [0, 14, 14],
    #     [14, 14, 0],
    #     [0, 0, 0]
    # ],
    # [
    #     [0, 0, 0, 0],
    #     [15, 15, 15, 15],
    #     [0, 0, 0, 0],
    #     [0, 0, 0, 0]
    # ],
    # [
    #     [16, 16, 0],
    #     [0, 16, 0],
    #     [0, 16, 0]
    # ],
    # [
    #     [0, 17, 17],
    #     [0, 17, 0],
    #     [0, 17, 0]
    # ]
]

w = 800
h = 600

surface = pygame.display.set_mode((w, h))

def draw_board():
    s = 20 # for sidelength of each tile
    g = 2 # for gaps between tiles
    hs = w//2 - s*len(board[0])//2 # for horizontal shift of board on surface
    vs = h//2 - s*len(board)//2 # for vertical shift of board on surface
    for i in range(len(board)):
        for j in range(len(board[0])):
            pygame.draw.rect(surface, (0, 0, 0), pygame.Rect(hs+s*j, vs+s*i, s, s))
            if board[i][j] == 0 and i < 20 and j > 2 and j < 13:
                pygame.draw.rect(surface, (50, 50, 50), pygame.Rect(hs+s*j, vs+s*i, s-g, s-g))
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
    global reward, line_clear_count
    for i in range(len(board)-4, -1, -1):
        line_saturated = True
        for j in range(len(board[0])-4, 2, -1):
            if board[i][j] == 0:
                line_saturated = False
        if line_saturated:
            # print('line clear')
            # time.sleep(1)
            reward += 1.0
            line_clear_count += 1
            # print('line_clear_count:', line_clear_count)
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
    global piece, x, y, generator, board, t0, game_over
    # if time.time() - t0 > 1:
    if True:
        t0 = time.time()
        remove_piece()
        x += 1
        place_piece()
        if collision():
            remove_piece()
            x -= 1
            if x <= x_init:
                # print('game over')
                game_over = True
                board = copy.deepcopy(board_init)
                generator = piece_generator()
                piece, x, y = next(generator), x_init, y_init
                place_piece()
                return 
            place_piece()
            piece, x, y = next(generator), x_init, y_init 
            place_piece()

generator = piece_generator()
board = copy.deepcopy(board_init)
prev_board = copy.deepcopy(board_init)
piece = next(generator)
x_init = 0
y_init = len(board[0])//2 - len(piece)//2
x = x_init
y = y_init
place_piece()
t0 = time.time()

# agent
np.random.seed(seed=1)
gamma = 0.99
reward = 0.0
game_over = False
game_count = 0
# epsilon = 1.0
line_clear_count = 0
learning_rate = 0.01
k = np.concatenate([np.asarray(board).flatten(), np.asarray(prev_board).flatten()]).shape[0]
print(k)
syn0 = (2*np.random.random((k,k)) - 1) * 0.01
syn1 = (2*np.random.random((k,k)) - 1) * 0.01
syn2 = (2*np.random.random((k,5)) - 1) * 0.01
def process_ai_input():
    global board, prev_board
    # X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
    # X = np.asarray(board).flatten() - np.asarray(prev_board).flatten() 
    X = np.concatenate([np.asarray(board).flatten(), np.asarray(prev_board).flatten()]) * 0.01
    prev_board = board
    # y = np.array([[0,1,1,0]]).T
    # syn0 = 2*np.random.random((3,4)) - 1
    # syn1 = 2*np.random.random((4,1)) - 1
    # for j in range(60000):
    l1 = 1/(1+np.exp(-(np.dot(X,syn0))))
    l2 = 1/(1+np.exp(-(np.dot(l1,syn1))))
    l3 = 1/(1+np.exp(-(np.dot(l2,syn2))))
    # l3 = 1/(1+np.exp(-(np.dot(l1,syn1))))
    # print(l2)
    move = np.argmax(l3)
    # epsilon = 1.0/(game_count+1.0)
    epsilon = 0.01
    if np.random.random() < epsilon:
        move = np.random.randint(5)
    # if game_count < 1:
    #     move = np.random.randint(5)
    if move == 0:
        move_left()
    if move == 1:
        move_right()
    if move == 2:
        move_rotate()
    if move == 3:
        move_down()
    if move == 4:
        move_drop()
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                pygame.quit()
                sys.exit()
        # l2_delta = (y - l2)*(l2*(1-l2))
        # l1_delta = l2_delta.dot(syn1.T) * (l1 * (1-l1))
        # syn1 += l1.T.dot(l2_delta)
        # syn0 += X.T.dot(l1_delta)
    return l1, l2, l3

def train(l1, l2, l3):
    global syn0, syn1, syn2, reward, game_over, game_count
    y = copy.deepcopy(l3)
    # X = np.asarray(board).flatten()
    X = np.concatenate([np.asarray(board).flatten(), np.asarray(prev_board).flatten()]) * 0.01
    l1_prime = 1/(1+np.exp(-(np.dot(X,syn0))))
    l2_prime = 1/(1+np.exp(-(np.dot(l1_prime,syn1))))
    l3_prime = 1/(1+np.exp(-(np.dot(l2_prime,syn2))))
    # print('l3_prime', l3_prime.shape)
    move_prime = np.argmax(l3_prime)
    # reward += (np.random.random() - 0.5) * 0.5
    if game_over:
        y[move_prime] = reward
    else:
        y[move_prime] = reward + gamma * l3_prime[move_prime]
    l3_delta = (y - l3)*(l3*(1-l3))
    l2_delta = l3_delta.dot(syn2.T) * (l2 * (1-l2))
    l1_delta = l2_delta.dot(syn1.T) * (l1 * (1-l1))
    syn2 += np.expand_dims(l2,0).T.dot(np.expand_dims(l3_delta,0)) * learning_rate
    # print(syn1.shape, l1.shape, l2_delta.shape)
    syn1 += np.expand_dims(l1,0).T.dot(np.expand_dims(l2_delta,0)) * learning_rate
    # print(syn0.shape, X.shape, l1_delta.shape)
    syn0 += np.expand_dims(X,0).T.dot(np.expand_dims(l1_delta,0)) * learning_rate
    if game_over:
        game_count += 1
        print('game_count:', game_count)
        print('line_clear_count:', line_clear_count)
    reward = 0.0
    game_over = False

while True:
    # process_input()
    l1, l2, l3 = process_ai_input()
    gravity()
    clear_line()
    draw_board()
    train(l1, l2, l3)
