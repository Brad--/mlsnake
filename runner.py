import neuralnetworksbylayer as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import copy

from board import Board
import plot
# %matplotlib inline # Uncomment this for the notebook
# Game Runner & Reinforcement Controller

validActionLetters = np.array(['U', 'D', 'L', 'R'])
validActions = np.array([0, 1, 2, 3])
boardSize = 5 # 5 x 5
epsilon = 1

def initialState(size):
    return Board(size)
    
def nextStateAndReinforce(board, action):
    # nextState
    temp = Board(5)
    temp.update(board.score, board.size, board.snake, board.apple, board.gameOver)
    board = temp

    deltaT = 0.1
    scoreBeforeMove = board.score
    distBeforeMove = board.getDistance()
    board.move(action)

    # reinforce
    if distBeforeMove <= board.getDistance():
        # r = .1 # Not enough
        r = .5 # moved towards the apple
    else:
        r = -.1 # moved away from the apple

    if board.score > scoreBeforeMove:
        r = 1 # got an apple

    if board.gameOver == True:
        r = -1

    return board, r

def policy(qnet, board, epsilon):
    if np.random.rand(1) < epsilon:
        actioni = np.random.randint(validActions.shape[0])
    else:
        inputs = np.hstack(( np.tile(board.getState(), 
            (validActions.shape[0], 1)), validActions.reshape((-1,1))))
        qs = qnet.use(inputs)
        actioni = np.argmax(qs)

    # print(validActions[actioni])
    return validActions[actioni]

def makeSamples(qnet, nSteps):
    samples = []
    state = initialState(boardSize)
    act = policy(qnet, state, epsilon)
    oldact = act
    for iStep in range(nSteps):
        newState, r = nextStateAndReinforce(state, act)
        newAct = policy(qnet, newState, epsilon)
		
        samples.append(state.getState() + [act, r] + newState.getState() + [newAct])

        # if newState.gameOver == True:
        #     return np.array(samples)

        state = newState
        oldact = act
        act = newAct
    return np.array(samples)

def run():
    epsilon = 1
    gamma = 0.999
    amp = 4
    nTrials = 300 * amp
    nStepsPerTrial = 500 * amp
    # nTrials = 300
    # nStepsPerTrial = 500
    # nTrials = 800
    # nStepsPerTrial = 1000
    nSCGIterations = 30
    finalEpsilon = 0.01
    epsilonDecay = np.exp(np.log(finalEpsilon)/(nTrials)) 

    # nh = [10, 10]
    # nh = [5,5]
    # Stats Exchange recommended the # hidden nodes equals the mean of the inputs and outputs
    nh = [4, 4]

    qnet = nn.NeuralNetwork([7] + nh + [1])
    bounds = (0, boardSize)
    adj = (0, 1)

    # Set each of the input ranges (There should be one for each X input)
    # qnet.setInputRanges(( (0, 40), bounds, bounds, bounds, bounds, (0, 4) ))
    # Boolean isAdjacentToTheWall for up, down, left, right, dist x, y
    qnet.setInputRanges(( adj, adj, adj, adj, bounds, bounds, (0, 3) ))

    epsilonTrace = np.zeros(nTrials)
    rtrace = np.zeros(nTrials)
    trial = 0
    for trial in range(nTrials):
        samples = makeSamples(qnet, nStepsPerTrial)

        ns = 6
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
        # if trial % (nTrials//10) == 0 or trial == nTrials-1:
        # print(rtrace)


    # plot.plotStatus(qnet, X, R, trial,epsilonTrace,rtrace, validActions, nextQ)
    # clear_output(wait=True)
    plot.displayGame(nextQ, boardSize)

run()