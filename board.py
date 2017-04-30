from random import randint

class Board:
	'Stores information about the game board, such as the snake and apple positions'
	SNAKE = "s"
	APPLE = "a"
	EMPTY = "."

	field = []
	snake = []
	score = 0
	apple = (-1, -1)
	gameOver = False
	size = 0

	# Create new Board
	def __init__(self, board):
		if isinstance(board, Board):
			self.initInProgress(board)
		else:
			self.size = board
			self.snake.append((0,0))
			self.emptyField()
			self.dropApple()
			self.updateField()

	def initInProgress(self, board) :
		self.size = size;
		self.field = board.field
		self.snake = board.snake
		self.score = board.score
		self.apple = board.apple

	def emptyField(self):
		self.field = [[self.EMPTY for w in range(self.size)] for h in range(self.size)]

	# Updates the field matrix to accurately hold snake & Apple
	def updateField(self):
		self.emptyField()
		self.field[self.apple[0]][self.apple[1]] = self.APPLE
		for piece in self.snake:
			self.field[piece[0]][piece[1]] = self.SNAKE

	# Update the apple variable
	def dropApple(self):
		done = False
		while not done:
			x = randint(0, self.size - 1)
			y = randint(0, self.size - 1)
			if self.field[x][y] == self.EMPTY:
				self.apple = (x, y)
				done = True

	# Move the snake given a direction.
	def move(self, dir):
		head = self.snake[0]
		moveTo = head

		dir = dir.upper()
		if dir == "L":
			if head[1] - 1 >= 0:
				moveTo = (moveTo[0], moveTo[1] - 1)
				self.checkShift(moveTo)
			else:
				self.gameOver()

		elif dir == "R":
			if head[1] + 1 < len(self.field[0]):
				moveTo = (moveTo[0], moveTo[1] + 1)
				self.checkShift(moveTo)
			else:
				self.gameOver()

		elif dir == "U":
			if head[0] - 1 >= 0:
				moveTo = (moveTo[0] - 1, moveTo[1])
				self.checkShift(moveTo)
			else:
				self.gameOver()

		elif dir == "D":
			if head[0] + 1 < len(self.field):
				moveTo = (moveTo[0] + 1, moveTo[1])
				self.checkShift(moveTo)
			else:
				self.gameOver()
		else:
			print("Invalid move direction: ", dir)

		if not self.gameOver:
			self.updateField()

	# Handles collisions w/ wall, snake, and apple
	def checkShift(self, loc):
		move = self.field[loc[0]][loc[1]]
		if move == self.EMPTY:
			self.shiftSnake(loc)
		elif move == self.APPLE:
			self.eatApple(loc)
		else:
			self.gameOver()

	def shiftSnake(self, loc):
		self.snake = [loc] + self.snake[0:len(self.snake) - 1]
		# self.snake.append(0, loc)
		# # Remove the back of the snake
		# self.snake.remove(self.snake[len(snake) - 1])

	def eatApple(self, loc):
		self.snake = [loc] + self.snake[:]
		# self.snake.append(0, loc)
		self.score += 1
		self.dropApple()

	def gameOver(self):
		gameOver = True

	def printBoard(self):
		print("Score: ", self.score)
		for x in range(self.size):
			row = ""
			for y in range(self.size):
				row += self.field[x][y] + " "
			print(row)