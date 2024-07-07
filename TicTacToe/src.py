import math
import numpy as np
import pygame
import sys

pygame.init()

White = (255, 255, 255)
Gray = (180, 180, 180)
Red = (255, 0, 0)
Green = (0, 255, 0)
Black = (0, 0, 0)

Width = 300
Height = 300
LineW = 5
BoardR = 3
BoardC = 3
Square = Width // BoardC
CircleR = Square // 3
CircleW = 15
CrossWidth = 25

board = np.zeros((BoardR, BoardC), dtype=int)

screen = pygame.display.set_mode((Width, Height))
pygame.display.set_caption("Tic Tac Toe")
screen.fill(Black)


def drawLines(color=White):
    for i in range(1, BoardR):
        pygame.draw.line(screen, color, (0, i * Square), (Width, i * Square), LineW)
        pygame.draw.line(screen, color, (i * Square, 0), (i * Square, Height), LineW)

def drawMoves(color=White):
    for r in range(BoardR):
        for c in range(BoardC):
            if board[r][c] == 1:
                pygame.draw.circle(screen, color, (int(c * Square + Square // 2)), (int(r * Square + Square // 2)), CircleR, CircleW)
            elif board[r][c] == 2:
                pygame.draw.line(screen, color, (c * Square + Square // 4, r * Square + Square // 4), (c * Square + Square * 3 // 4, r * Square + Square * 3 // 4), CrossWidth)
                pygame.draw.line(screen, color, (c * Square + Square * 3 // 4, r * Square + Square // 4), (c * Square + Square // 4, r * Square + Square * 3 // 4), CrossWidth)

def markSquare(row, col, player):
    board[row][col] = player

def square_is_Avaliable(row, col):
    return board[row][col] == 0


def BoardisFull(check_board=board):
    for r in range(BoardR):
        for c in range(BoardC):
            if board[r][c] == 0:
                return False
    return True


def winning(player, check_board=board):
    for c in range(BoardC):
        if (check_board[0][c] == player and check_board[1][c] == player and check_board[2][c] == player) or (check_board[c][0] == player and check_board[c][1] == player and check_board[c][2] == player):
            return True
        
    for r in range(BoardR):
        if (check_board[r][0] == player and check_board[r][1] == player and check_board[r][2] == player) or (check_board[0][r] == player and check_board[1][r] == player and check_board[2][r] == player):
            return True
        
    if (check_board[0][0] == player and check_board[1][1] == player and check_board[2][2] == player) or (check_board[0][2] == player and check_board[1][1] == player and check_board[2][0] == player):
        return True
    
    return False



def minimax(minimaxBoard, depth, maximizing):
    if winning(2, minimaxBoard):
        return float('inf')
    elif winning(1, minimaxBoard):
        return float('-inf')
    elif BoardisFull(minimaxBoard):
        return 0 

    if maximizing:
        BestScore = -1000
        for r in range(BoardR):
            for c in range(BoardC):
                if minimaxBoard[r][c] == 0:
                    minimaxBoard[r][c] = 2
                    score = minimax(minimaxBoard, depth + 1, False)
                    minimaxBoard[r][c] = 0
                    BestScore = max(score, BestScore)
        return BestScore
    else:
        BestScore = 1000
        for r in range(BoardR):
            for c in range(BoardC):
                if minimaxBoard[r][c] == 0:
                    minimaxBoard[r][c] = 1
                    score = minimax(minimaxBoard, depth + 1, True)
                    minimaxBoard[r][c] = 0
                    BestScore = min(score, BestScore)
        return BestScore
    


def BestMove():
    BestScore = -1000
    move = (-1, -1)
    for r in range(BoardR):
        for c in range(BoardC):
            if board[r][c] == 0:
                board[r][c] = 2
                score = minimax(board, 0, False)
                board[r][c] = 0
                if score > BestScore:
                    BestScore = score
                    move = (r, c)

    if move != (-1, -1):
        markSquare(move[0], move[1], 2)
        return True
    return False                    


        
    
def restart():
    screen.fill(Black)
    drawLines()
    for r in range(BoardR):
        for c in range(BoardC):
            board[r][c] = 0


player = 1
gameOver = False

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

        if event.type == pygame.MOUSEBUTTONDOWN and not gameOver:
            mouseX = event.pos[0] // Square
            mouseY = event.pos[1] // Square

            if square_is_Avaliable(mouseY, mouseX):
                markSquare(mouseY, mouseX, player)
                if winning(player):
                    gameOver = True
                    player = player % 2 + 1

                if not gameOver:
                    if BestMove():
                        if winning(2):
                            gameOver = True
                    player = player % 2 + 1

                if not gameOver:
                    if BoardisFull():
                       gameOver = True

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                restart()
                gameOver = False
                player = 1

    if not gameOver:
        drawMoves()
        pygame.display.update()   
    else:
        if winning(1):
            drawMoves(Green)
            drawLines(Green)
