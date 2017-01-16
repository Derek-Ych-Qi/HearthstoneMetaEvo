import numpy as np


def GenRandomWinrateMatrix(N_CLASS):
    """
    Generate a random winrate matrix with N_CLASS number of classes
    """
    winrate = np.eye(N_CLASS) * 0.5
    for ri in range(N_CLASS):
        for ci in range(ri):
            winrate[ri, ci] = np.random.random()

    for ri in range(N_CLASS):
        for ci in range(ri + 1, N_CLASS):
            winrate[ri, ci] = 1 - winrate[ci, ri]

    return winrate


def GenRandomMeta(N_CLASS):
    """
    Generate a random meta with N_CLASS number of classes.
    """
    meta = np.random.random(N_CLASS)
    meta /= sum(meta)
    return meta

if __name__ == "__main__":
    m = GenRandomMeta(3)
    print(m)