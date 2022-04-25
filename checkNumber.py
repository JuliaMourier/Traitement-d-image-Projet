import numpy as np


def checkSudoku(sudoku) :
    isGood = False

    while not isGood :
        print("Is the sudoku good for you ? y/(n)")
        isGoodStr = input()
        if isGoodStr == "y" or isGoodStr == "Y":
            isGood = True
            continue

        print("What number do you want to change ?")
        print("row (from 0 to 8) :")
        row = int(input())
        if row > 8 or row < 0 or not type(row) == int:
            print("incorrect number of row, should be between 0 and 8")
            continue

        print("row (from 0 to 8) :")
        column = int(input())
        if column > 8 or column < 0 or not type(column) == int :
            print("incorrect number of column, should be between 0 and 8")
            continue

        print("The value you want to change is : " + str(sudoku[row][column]))
        print("Give the value you want : (from 1 to 9)")
        changed_value = int(input())
        if changed_value < 0 or changed_value > 9 or not type(changed_value) == int:
            print("incorrect value, should be between 1 and 9")
        else :
            sudoku[row][column] = changed_value

        print("\nSudoku :")
        print(sudoku)

    return sudoku

#sudoku = np.zeros((9,9))
#print(sudoku)
#checkSudoku(sudoku)

