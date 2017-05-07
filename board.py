from random import randint
import math

class Board:
	'Stores information about the game board, such as the snake and apple positions'
	SNAKE = "s"
	APPLE = "a"
	EMPTY = "."
	OVER = "x"

	field = []
	snake = []
	score = 0
	apple = (-1, -1)
	gameOver = False
	size = 0

	# Create new Board
	def __init__(self, board):
		self.size = board
		mid = math.floor(self.size / 2)
		self.emptyField()
		self.snake = [(mid, mid)]
		self.field[mid][mid] = self.SNAKE 
		self.dropApple()
		self.updateField()

	def update(self, score, size, snake, apple, gameOver):
		self.score = score
		self.size = size
		self.snake = snake
		self.apple = apple
		self.gameOver = gameOver
		self.updateField()

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

		self.updateField()

	# Move the snake given a direction.
	def move(self, dir):
		head = self.snake[0]
		moveTo = head

		#Up
		if dir == 0:
			if head[0] - 1 >= 0:
				moveTo = (moveTo[0] - 1, moveTo[1])
				self.checkShift(moveTo)
			else:
				self.lose()
		#Down
		elif dir == 1:
			if head[0] + 1 < self.size:
				moveTo = (moveTo[0] + 1, moveTo[1])
				self.checkShift(moveTo)
			else:
				self.lose()
		#Left
		elif dir == 2 :
			if head[1] - 1 >= 0:
				moveTo = (moveTo[0], moveTo[1] - 1)
				self.checkShift(moveTo)
			else:
				self.lose()
		#Right
		elif dir == 3:
			if head[1] + 1 < self.size:
				moveTo = (moveTo[0], moveTo[1] + 1)
				self.checkShift(moveTo)
			else:
				self.lose()

		

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
			self.lose()

	def shiftSnake(self, loc):
		self.snake = [loc] + self.snake[:len(self.snake) - 1]
		self.updateField()

	def eatApple(self, loc):
		self.snake = [loc] + self.snake[:len(self.snake) - 1] # Don't extend the snake because it doesn't seem to be very good at adapting
		self.score += 1
		self.dropApple()

	def lose(self):
		# print("Game Over")
		self.gameOver = True

	def getState(self):
		# return [self.score] + list(self.apple) + self.snake
		#Up down left right
		if self.snake[0][0] == 0:
			adjTop = 1
		else:
			adjTop = 0

		if self.snake[0][0] == self.size - 1:
			adjBot = 1
		else:
			adjBot = 0

		if self.snake[0][1] == 0:
			adjLeft = 1
		else:
			adjLeft = 0

		if self.snake[0][1] == self.size - 1:
			adjRight = 1
		else:
			adjRight = 0


		# Boolean isAdjacentToTheWall for up, down, left, right, dist x, y

		# x, y = self.getDistanceXY()
		return [adjTop, adjBot, adjLeft, adjRight]

	# Minimum number of moves from the head of the snake to the apple
	def getDistance(self):
		return abs(self.apple[0] - self.snake[0][0]) + abs(self.apple[1] - self.snake[0][1])

	def getDistanceXY(self):
		return abs(self.apple[0] - self.snake[0][0]), abs(self.apple[1] - self.snake[0][1])

	def printBoard(self):
		print("Score: ", self.score)
		if self.gameOver == True:
			print("Game Over.")
		for x in range(self.size):
			row = ""
			for y in range(self.size):
				row += self.field[x][y] + " "
			print(row)