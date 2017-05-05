import matplotlib.pyplot as plt
import numpy as np
import sys
from board import Board

from IPython.display import display, clear_output
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

validActions = np.array([0, 1, 2, 3])
validActionLetters = np.array(['U', 'D', 'L', 'R'])

def displayGame(qs, boardSize):
    actsiByState = np.argmax(qs.reshape((len(validActions),-1)),axis=0)
    
    # Print to 'out.txt' instead of console
    f = open('out.txt', 'w')
    sys.stdout = f

    board = Board(boardSize)
    board.printBoard()
    print()

    # Print "Ideal game"
    for i in range(40):
        if not board.gameOver:
            print("Move: ", validActionLetters[actsiByState[i]])
            board.move(actsiByState[i])
            board.printBoard()
            print()

    # Print just the moves made
    for i in range(4):
        row = ""
        for j in range(10):
            row += str(validActionLetters[actsiByState[(i * 10) + j ] ]) + ", "
        print(row)

def plotStatus(qnet, X, R, trial, epsilonTrace, rtrace, qs):
    fig = plt.figure(figsize=(15,15))

    plt.subplot(4,3,1)
    plt.plot(epsilonTrace[:trial+1])
    plt.ylabel("Random Action Probability ($\epsilon$)")
    plt.ylim(0,1)

    numActions = 40
    plt.subplot(4,3,2)
    plt.plot(X[:,0])
    plt.plot([0,X.shape[0]], [5,5],'--',alpha=0.5,lw=5)
    plt.ylabel("$x$")
    plt.ylim(-1,11)
    # qs = qnet.use(np.array([[s,0,0,0,0,a] for a in validActions 
    #     for s in range(numActions)]))
    #print np.hstack((qs,-1+np.argmax(qs,axis=1).reshape((-1,1))))
    
    plt.subplot(4,3,3)
    acts = ["U", "D", "L", "R"]
    actsiByState = np.argmax(qs.reshape((len(validActions),-1)),axis=0)
    
    # unique = False
    # act = actsiByState[0]
    # for i in range(len(actsiByState)):
    #     if actsiByState[i] != act:
    #         unique = True

    # if not unique:
    #     return -1

    for i in range(numActions):
        if i < 10:
            j = 0
        elif i < 20:
            j = -.25
        elif i < 30:
            j = -.5
        else:
            j = -.75

        plt.text(i % 10,j,acts[actsiByState[i]])
        plt.xlim(-1,11)
        plt.ylim(-1,1)
    plt.text(2,0.2,"Policy")
    plt.axis("off")
    
    plt.subplot(4,3,4)
    plt.plot(rtrace[:trial+1],alpha=0.5)
    #plt.plot(np.convolve(rtrace[:trial+1],np.array([0.02]*50),mode='valid'))
    binSize = 20
    if trial+1 > binSize:
        # Calculate mean of every bin of binSize reinforcement values
        smoothed = np.mean(rtrace[:int(trial/binSize)*binSize].reshape((int(trial/binSize),binSize)),axis=1)
        plt.plot(np.arange(1,1+int(trial/binSize))*binSize,smoothed)
    plt.ylabel("Mean reinforcement")
    
    plt.subplot(4,3,5)
    plt.plot(X[:,0],X[:,1])
    plt.plot(X[0,0],X[0,1],'o')
    plt.xlabel("$x$")
    plt.ylabel("$\dot{x}$")
    plt.fill_between([4,6],[-5,-5],[5,5],color="red",alpha=0.3)
    plt.xlim(-1,11)
    plt.ylim(-5,5)
    
    plt.subplot(4,3,6)
    qnet.draw(["$x$","$\dot{x}$","$a$"],["Q"])

    # plt.subplot(4,3,7)
    # n = 20
    # positions = np.linspace(0,10,n)
    # velocities =  np.linspace(-5,5,n)
    # xs,ys = np.meshgrid(positions,velocities)
    # #states = np.vstack((xs.flat,ys.flat)).T
    # #qs = [qnet.use(np.hstack((states,np.ones((states.shape[0],1))*act))) for act in actions]
    # xsflat = xs.flat
    # ysflat = ys.flat
    # qs = qnet.use(np.array([[xsflat[i],ysflat[i],a] for a in validActions for i in range(len(xsflat))]))
    # #qs = np.array(qs).squeeze().T
    # qs = qs.reshape((len(validActions),-1)).T
    # qsmax = np.max(qs,axis=1).reshape(xs.shape)
    # cs = plt.contourf(xs,ys,qsmax)
    # plt.colorbar(cs)
    # plt.xlabel("$x$")
    # plt.ylabel("$\dot{x}$")
    # plt.title("Max Q")
    
    # plt.subplot(4,3,8)
    # acts = np.array(validActions)[np.argmax(qs,axis=1)].reshape(xs.shape)
    # cs = plt.contourf(xs,ys,acts,[-2, -0.5, 0.5, 2])
    # plt.colorbar(cs)
    # plt.xlabel("$x$")
    # plt.ylabel("$\dot{x}$")
    # plt.title("Actions")

    # s = plt.subplot(4,3,10)
    # rect = s.get_position()
    # ax = Axes3D(plt.gcf(),rect=rect)
    # ax.plot_surface(xs,ys,qsmax,cstride=1,rstride=1,cmap=cm.viridis,linewidth=0)
    # ax.set_xlabel("$x$")
    # ax.set_ylabel("$\dot{x}$")
    # #ax.set_zlabel("Max Q")
    # plt.title("Max Q")

    # s = plt.subplot(4,3,11)
    # rect = s.get_position()
    # ax = Axes3D(plt.gcf(),rect=rect)
    # ax.plot_surface(xs,ys,acts,cstride=1,rstride=1,cmap=cm.viridis,linewidth=0)
    # ax.set_xlabel("$x$")
    # ax.set_ylabel("$\dot{x}$")
    # #ax.set_zlabel("Action")
    # plt.title("Action")

    # display(fig);
    plt.show()
    return 0