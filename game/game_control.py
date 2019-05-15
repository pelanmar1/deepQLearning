import pyautogui
import time 

import random
import game.browser_control

currentGrid = [[0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0]]

UP = 100
LEFT = 101
DOWN = 102
RIGHT = 103
SCALE_FACTOR = 2
SPEED = 9

SPEED_LIST = list(reversed([0.00001,0.00005,0.0005,0.005,0.01,0.05,0.1,0.5,1,2]))
WAIT = SPEED_LIST[SPEED]


def moves_map(move_id):
        # 0, 1, 2, 3 are up, right, down, left
        # moves_map = {0:UP,1:DOWN,2:LEFT,3:RIGHT}
        moves_map = {0:UP,1:RIGHT,2:DOWN,3:LEFT}
        return moves_map[move_id]

def getGrid():
    currentGrid = game.browser_control.get_grid()
    return currentGrid
    
def printGrid(grid):
    for t in range(len(grid)):
        print(grid[t])

def performMove(action):
    move = moves_map(action)
    pyautogui.click(x=65,y=266)
    if move == UP:
        pyautogui.keyDown('up')
        # time.sleep(0.05)
        pyautogui.keyUp('up')
    elif move == DOWN:
        pyautogui.keyDown('down')
        # time.sleep(0.05)
        pyautogui.keyUp('down')
    elif move == LEFT:
        pyautogui.keyDown('left')
        # time.sleep(0.05)
        pyautogui.keyUp('left')
    else:
        pyautogui.keyDown('right')
        # time.sleep(0.05)
        pyautogui.keyUp('right')
    # pyautogui.click(x=1200*2,y=850*2)
    pyautogui.PAUSE = WAIT
    # time.sleep(0.3)
    
def restart_game():
    print("STARTING NEW GAME")
    # time.sleep(0.5)
    pyautogui.moveTo(x=530,y=265)
    pyautogui.click(x=530,y=265)
    pyautogui.click(x=530,y=265)
    # time.sleep(0.5)
    pyautogui.click(x=1120,y=500)
    # time.sleep(0.5)

def getScore():
    score = game.browser_control.get_score()      
    return score

def getBestScore():
    bestScore = game.browser_control.get_best_score()
    return bestScore

def isOver():
    isOver = game.browser_control.is_over()
    return isOver
 