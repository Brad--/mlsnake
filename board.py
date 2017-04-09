from random import randint

class Board:
	'Stores information about the game board, such as the snake and apple positions'
	SNAKE = "s"
	APPLE = "a"
	EMPTY = "."

	snake = []
	score = 0
	apple = (-1, -1)
	gameOver = False

	# Create new Board
	def __init__(self, size):
		gameOver = False
		snake.push((0,0))
		dropApple()
		updateField()

	def __init__(self, board):
		self.field = board.field
		self.snake = board.snake
		self.score = board.score
		self.apple = board.apple

	def emptyField():
		field = [[EMPTY for w in range(size)] for h in range(size)]

	# Updates the field matrix to accurately hold snake & Apple
	def updateField():
		emptyField()
		field[apple[0]][apple[1]] = APPLE
		for piece in snake:
			field[piece[0]][piece[1]] = SNAKE

	# Update the apple variable
	def dropApple():
		done = False
		while not done:
			x = randint(len(field))
			y = randint(len(field[0]))
			if field[x][y] == EMPTY:
				apple = (x, y)
				done = True

	# Move the snake given a direction.
	def move(dir):
		head = snake[0]
		moveTo = head
		if dir == "LEFT":
			if head[1] - 1 >= 0:
				moveTo[1] -= 1
				checkShift(moveTo)
			else:
				gameOver()

		elif dir == "RIGHT":
			if head[1] + 1 < len(field[0]):
				moveTo[1] += 1
				checkShift(moveTo)
			else:
				gameOver()

		elif dir == "UP":
			if head[0] - 1 >= 0:
				moveTo[0] -= 1
				checkShift(moveTo)
			else:
				gameOver()

		elif dir == "DOWN":
			if head[0] + 1 < len(field):
				moveTo[0] += 1
				checkShift(moveTo)
			else:
				gameOver()
		else:
			print("Invalid move direction: ", dir)

		if not gameOver:
			updateField()

	# Handles collisions w/ wall, snake, and apple
	def checkShift(loc):
		move = board[loc[0]][loc[1]]
		if move == EMPTY:
			shiftSnake(loc)
		elif move == APPLE:
			eatApple(loc)
		else:
			gameOver()

	def shiftSnake(loc):
		snake.append(0, loc)
		# Remove the back of the snake
		snake.remove(snake[snake(len) - 1])

	def eatApple(loc):
		snake.append(0, loc)
		score += 1
		dropApple()

	# Intentionally leaves the cause vague so the machine can learn it itself
	def gameOver():
		gameOver = True

	def printBoard():
		dim = len(field)
		print("Score: ", score)
		for x in range(dim):
			row = ""
			for y in range(dim):
				row += field[x][y]
			print(row)