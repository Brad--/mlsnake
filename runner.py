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
# boardSize = 3
# boardSize = 4
boardSize = 5
# boardSize = 10
epsilon = 1

def initialState(size):
    return Board(size)
    
def nextStateAndReinforce(board, action):
    # nextState
    temp = Board(boardSize)
    temp.update(board.score, board.size, board.snake, board.apple, board.gameOver)
    board = temp

    scoreBeforeMove = board.score
    gameOverBeforeMove = board.gameOver
    distBeforeMove = board.getDistance()
    board.move(action)

    # Trying the Stanford paper numbers
    # if board.gameOver == True and board.gameOver != gameOverBeforeMove:
    #     r = -100
    # elif board.score > scoreBeforeMove:
    #     r = 500
    # # elif board.gameOver == True and board.gameOver == gameOverBeforeMove:
    # #     r = 0 # Nothing changed, the game was over
    # else:
    #     r = -10 # More moves is bad so punish non-scoring moves

    # The 'Just Don't Die' method
    if board.gameOver == True:
        r = -10
    elif board.score > scoreBeforeMove:
        r = 100 # Get the apple
    else:
        r = 10

    # Original method
    # if board.score > scoreBeforeMove:
    #     r = 100 # got an apple
    #     # r = 1000
    # elif distBeforeMove <= board.getDistance():
    #     # r = 500
    #     r = 10
    #     # r = 100
    # else:
    #     r = -10
    # if board.gameOver != gameOverBeforeMove:
    #     # r = -5000
    #     r = -500

    # # I'm not sure if this should be 0, or just let it fall into -10.\
    # if board.gameOver == True and board.gameOver == gameOverBeforeMove:
    #     r = 0
    #     # r = -1

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
    # gamma = 1
    gamma = 0.999 # Nothing consistent with the other two
    # gamma = 0.1
    
    amp = 1 # Attempts: 4, 10, 4, 16
    # amp = 2 # Try a middle ground. Attempts: 6, 6, 6
    # amp = 10 # Attempts: 6, 6, 6 (Works really well with the stanford method)
    # amp = 1000
    # amp = 10000
    # amp = 100 # After reducing nStepsPerTrial to a reasonable amount of moves the game could make
    # nTrials = 100 * amp
    # nTrials = 100 * amp
    # nStepsPerTrial = 40
    # nStepsPerTrial = 1000
    # nStepsPerTrial = 100

    # I'm pretty sure this is the variable I've been meaning to tinker with the whole time tbh
    # nSCGIterations = 10
    nSCGIterations = 100
    # nSCGIterations = 1000 # This works really well
    # nSCGIterations = 10000 # mask off
    # nSCGIterations = 100000 

    # Attempts: 
    # amp = 20
    # amp = 4
    # nTrials = 300 * amp
    # nStepsPerTrial = nTrials # 500 moves takes a long time and is unreasonable


    nTrials = 800
    nStepsPerTrial = 250
    # nStepsPerTrial = 500
    # nStepsPerTrial = 1000

    # nSCGIterations = 30
    # nSCGIterations = 100
    finalEpsilon = 0.01
    # finalEpsilon = 0.001
    # finalEpsilon = .1
    epsilonDecay = np.exp(np.log(finalEpsilon)/(nTrials)) 

    nh = [10, 10] # Attempts: 11 moves, 7 moves, 8 moves
    # nh = [5,5]
    # Stats Exchange recommended the # hidden nodes equals the mean of the inputs and outputs
    # nh = [4, 4] # Attempts: 4, 8, 5

    qnet = nn.NeuralNetwork([5] + nh + [1])
    bounds = (0, boardSize)
    adj = (0, 1)

    # Set each of the input ranges (There should be one for each X input)
    # qnet.setInputRanges(( (0, 40), bounds, bounds, bounds, bounds, (0, 4) ))

    # isAdjacentToTheWall for up, down, left, right, dist x, y
    qnet.setInputRanges(( adj, adj, adj, adj, (0, 3) ))

    epsilonTrace = np.zeros(nTrials)
    rtrace = np.zeros(nTrials)
    trial = 0
    for trial in range(nTrials):
        if trial % 10 == 0:
            print("Running trial " + str(trial), flush=True)
        samples = makeSamples(qnet, nStepsPerTrial)

        ns = 4
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
    # qs = qnet.use(np.array([[s,0,0,0,0,0,a] 
    #     for a in validActions for s in range(11)]))
    qUse = [[s1, s2, s3, s4, a] for a in validActions 
    for s1 in range(2) for s2 in range(2) for s3 in range(2) for s4 in range(2)]
    qs = qnet.use(np.array(qUse))
    plot.displayGame(qs, boardSize)
    # plot.displayGame(nextQ, boardSize)

run()