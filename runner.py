import neuralnetworksbylayer as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import copy

from board import Board
# Game Runner & Reinforcement Controller

validActions = np.array(['U', 'D', 'L', 'R'])
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
		
		samples.append([state.score] + list(state.apple) + state.snake + [act, r] 
			+ [newState.score] + list(newState.apple) + newState.snake + [newAct])

		state = newState
		oldact = act
		act = newAct
	return np.array(samples)

def run():
	gamma = 0.999
	nTrials = 300
	nStepsPerTrial = 500 
	nSCGIterations = 30
	finalEpsilon = 0.01
	epsilonDecay = np.exp(np.log(finalEpsilon)/(nTrials)) 

	nh = [5,5]
	qnet = nn.NeuralNetwork([3] + nh + [1])
	qnet.setInputRanges(( (0, 10), (-3, 3), (-1,1)))
	
	fig = plt.figure(figsize=(15,15))

	epsilonTrace = np.zeros(nTrials)
	rtrace = np.zeros(nTrials)

	#End Variable Initialization block

	for trial in range(nTrials):
	    samples = makeSamples(qnet, nStepsPerTrial)

	    ns = 2
	    na = 1
	    X = samples[:, :ns+na]
	    R = samples[:, ns+na:ns+na+1]
	    nextX = samples[:, ns+na+1:]
	    nextQ = qnet.use(nextX)

	    qnet.train(X, R + gamma * nextQ, nIterations = nSCGIterations)
	    
	    epsilon *= epsilonDecay

	    # Rest is for plotting
	    epsilonTrace[trial] = epsilon
	    rtrace[trial] = np.mean(R)

	    if trial % (nTrials//10) == 0 or trial == nTrials-1:
	        plt.clf()
	        plotStatus(qnet, X, R, trial,epsilonTrace,rtrace)
	        testIt(qnet,10,500)
	        clear_output(wait=True)
	        display(fig);
	        plt.pause(0.01)

	    # print('Trial',trial,'mean R',np.mean(R))
	clear_output(wait=True)

run()