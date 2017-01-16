import numpy as np
import RandomdataGen as rdg


def solve_meta_eqm(winrate):
    """
    Solve for the meta equilibrium given a win rate matrix
    """

    N_CLASS = winrate.shape[0]
    eqm_meta = np.zeros([N_CLASS, 1])
    eqm_meta = np.dot(np.linalg.inv(winrate_mat), np.array([0.5] * N_CLASS))
    old_eqm_meta = eqm_meta.copy()
    new_eqm_meta = eqm_meta.copy()

    eqm_nonzero = new_eqm_meta[np.nonzero(new_eqm_meta)]
    while min(eqm_nonzero) < 0:
        selected = np.where(new_eqm_meta > 0)[0]
        N_SEL = len(selected)
        winrate_selected = winrate[np.ix_(selected, selected)]
        old_eqm_meta = new_eqm_meta.copy()
        temp = np.dot(np.linalg.inv(winrate_selected), np.array([0.5] * N_SEL))
        deleted = np.where(new_eqm_meta <= 0)[0]
        new_eqm_meta[deleted] = 0
        new_eqm_meta[selected] = temp

        eqm_nonzero = new_eqm_meta[np.nonzero(new_eqm_meta)]
        if eqm_nonzero.shape[0] > 1:
            print(new_eqm_meta)
    return new_eqm_meta


if __name__ == "__main__":
    # load Meta data and Win rate data
    # cur_meta = np.loadtxt('SampleMeta.csv')
    # winrate_mat = np.loadtxt('SampleWinrate.csv', delimiter=',')

    # use random generator
    np.random.seed(123)
    N_CLASS = 6
    winrate_mat = rdg.GenRandomWinrateMatrix(N_CLASS)
    cur_meta = rdg.GenRandomMeta(N_CLASS)

    eqm_meta = solve_meta_eqm(winrate_mat)
    print(eqm_meta)