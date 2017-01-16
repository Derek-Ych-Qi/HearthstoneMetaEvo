import numpy as np


if __name__ == "__main__":
    # load Meta data and Win rate data
    cur_meta = np.loadtxt('SampleMeta.csv')
    winrate_mat = np.loadtxt('SampleWinrate.csv', delimiter=',')

    N_CLASS = int(cur_meta.shape[0])
    cur_meta = cur_meta.reshape([N_CLASS, 1])

    ewinrate = np.dot(winrate_mat, cur_meta)
    print(ewinrate)

    eqm_meta = np.dot(np.linalg.inv(winrate_mat), np.array([0.5] * N_CLASS)) # equilibrium meta
    print(eqm_meta)