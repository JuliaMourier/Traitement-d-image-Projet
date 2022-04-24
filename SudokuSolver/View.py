import tkinter as tk

class View:
    # Take a window and a sudoku
    def __init__(self, sudoku: []):
        self.win = tk.Tk()

        self.win.title("Sudoku Solveur")
        self.sudoku = sudoku

        # Create a grid 3x3 of grids 3x3
        for y in range(0,3):
            for x in range(0,3):
                tk.Frame(self.win, highlightthickness=3, highlightbackground="black", width=80*3, height=80*3).grid(row=y, column=x)
                frame = self.win.grid_slaves(y, x)[0]
                for j in range(0,3):
                    for i in range(0,3):
                        tk.Canvas(frame, highlightthickness=2, highlightbackground="black", width=80,height=80).grid(row=j, column=i)

        # Wait 10ms and fill the canvas with the sudoku's values
        self.win.after(10, self.fill_canvas())
        self.win.mainloop()

    # Fill the canvas with the sudoku's values
    def fill_canvas(self):
        # Creation of 3x3 frames
        for y in range(0, 3):
            for x in range(0, 3):
                frame = self.win.grid_slaves(y, x)[0]
                # Create a frame for each sqaure
                for j in range(0,3):
                    for i in range(0,3):
                        can = frame.grid_slaves(j,i)[0]
                        can.delete("all")
                        # If the square is not empty
                        if self.sudoku[y*3+j][x*3+i] != 0:
                            text = str(self.sudoku[y*3+j][x*3+i])
                            # Add the number of the square of the sudoku if it is not empty at the right place
                            can.create_text(40, 40, text=text, fill="black", font=('Helvetica 30 bold'))
