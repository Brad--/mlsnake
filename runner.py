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

# http://cs229.stanford.edu/proj2016spr/report/060.pdf
# Lots of cool stuff ^

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
    gameOverBeforeMove = board.gameOver
    distBeforeMove = board.getDistance()
    board.move(action)

    # # reinforce
    # if distBeforeMove <= board.getDistance():
    #     # r = .1 # Not enough
    #     # r = 1 # moved towards the apple
    #     # r = 10 # beef it up 8, 8, 6
    #     r = 
    # else:
    #     # r = -.1 # moved away from the apple
    #     # r = -.2 # beef it up? down?
    #     r = -.4 # More and or less beef. Attempts: 11 moves, 7 moves, 8 moves (With 10, 10)

    # if board.score > scoreBeforeMove:
    #     # r = 1 # got an apple
    #     r = 10

    # if board.gameOver == True:
    #     r = -1
    #     # r = -10 # No noticable difference between -1 & 10 tbh

    # Trying the Stanford paper numbers
    # 6, 
    if board.gameOver != gameOverBeforeMove:
        r = -100
    elif board.score > scoreBeforeMove:
        r = 500
    elif board.gameOver == True and board.gameOver == gameOverBeforeMove:
        r = 0 # Nothing changed, the game was over
    else:
        r = -10 # More moves is bad so punish bad moves

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
    # amp = 4 # Attempts: 6, 6, 6 (Works really well with the stanford method)
    # amp = 10 # mask off (This one actually sucked)
    
    # amp = 1 # Attempts: 4, 10, 4, 16
    # amp = 2 # Try a middle ground. Attempts: 6, 6, 6
    
    amp = 10 # After reducing nStepsPerTrial to a reasonable amount of moves the game could make
    nTrials = 300 * amp
    nStepsPerTrial = 40

    # Attempts: 
    # amp = 20
    # amp = 4
    # nTrials = 300 * amp
    # nStepsPerTrial = nTrials # 500 moves takes a long time and is unreasonable


    # nTrials = 800
    # nStepsPerTrial = 1000

    nSCGIterations = 30
    # nSCGIterations = 30
    # nSCGIterations = 300 # 6, 4 
    finalEpsilon = 0.01
    epsilonDecay = np.exp(np.log(finalEpsilon)/(nTrials)) 

    nh = [10, 10] # Attempts: 11 moves, 7 moves, 8 moves
    # nh = [5,5]
    # Stats Exchange recommended the # hidden nodes equals the mean of the inputs and outputs
    # nh = [4, 4] # Attempts: 4, 8, 5

    qnet = nn.NeuralNetwork([7] + nh + [1])
    bounds = (0, boardSize)
    adj = (0, 1)

    # Set each of the input ranges (There should be one for each X input)
    # qnet.setInputRanges(( (0, 40), bounds, bounds, bounds, bounds, (0, 4) ))

    # isAdjacentToTheWall for up, down, left, right, dist x, y
    qnet.setInputRanges(( adj, adj, adj, adj, bounds, bounds, (0, 3) ))

    epsilonTrace = np.zeros(nTrials)
    rtrace = np.zeros(nTrials)
    trial = 0
    for trial in range(nTrials):
        if trial % 100 == 0:
            print("Running trial " + str(trial), flush=True)
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