import neuralnetworksbylayer as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import copy

from board import Board
# Game Runner & Reinforcement Controller
def init():
	validActions = ['U', 'D', 'L', 'R']
	boardSize = 10
	epsilon = 1

def initialState(size):
	return Board(size)

def nextStateAndReinforce(board, action):
	# nextState
	board = copy.copy(board)
	deltaT = 0.1
	scoreBeforeMove = board.score
	board.move(action)
	# reinforce
	if board.score == scoreBeforeMove:
		r = -.01
	else:
		r = 0

	if board.gameOver:
		r = -1
	return board, r

def policy(qnet, board, epsilon):
	if np.random.rand(1) < epsilon:
		actioni = np.random.randint(validActions.shape[0])
	else:
		# This might be really wrong, I haven't thought it through yet
		# inputs = np.hstack(( np.tile(state, (validActions.shape[0], 1)), validActions.reshape((-1,1))))
		inputs = np.hstack(( np.tile(board.score, (validActions.shape[0], 1)), validActions.reshape((-1,1))))
		qs = qnet.use(inputs)
		actioni = np.argmax(qs)
	return validActions[actioni]

def makeSamples(qnet, nSteps):
	samples = []
	state = initialState(boardSize)
	act = policy(qnet, state, epsilon)
	oldact = act
	for iStep in range(nSteps):
		newState, r = nextStateAndReinforce(state, act)
		newAct = policy(qnet, newState, epsilon)
		# Not sure about state.score
		samples.append(state.score + [act, r] + newState.score + [newAct])
		state = newState
		oldact = act
		act = newAct
	return np.array(samples)

def run():
	init()

run()