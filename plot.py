import matplotlib.pyplot as plt

def MakePLT(x, y, X, Y, L, path):
    plt.figure()
    plt.plot(x,y)
    plt.xlabel(X)
    plt.ylabel(Y)
    if L!=None:
        plt.legend(L)
    plt.savefig(path)
    return