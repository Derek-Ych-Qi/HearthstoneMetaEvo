import numpy as np
import RandomdataGen as rdg

def meta_evo(winrate_mat, cur_meta, tol):
    """
    Iteratively solve for equilibrium status with win rate matrix given
    :param winrate_mat: win rate matrix
    :param cur_meta: current meta
    :param tol: tolerance
    :return: The final equilibrium meta
    """
    ic = 0
    threshold = 1e-9
    ewinrate = np.dot(winrate_mat, cur_meta)
    dif = ewinrate.max() - ewinrate.min()
    new_meta = cur_meta.copy()
    while dif > tol:
        old_meta = new_meta.copy()
        # dmeta = ewinrate - np.mean(ewinrate)
        # dmeta /= np.linalg.norm(dmeta) * delta
        # new_meta += dmeta

        new_meta[np.where(ewinrate == ewinrate.max())] += dif * 10 ** -2
        new_meta[np.where(new_meta < threshold)] = 0
        new_meta /= sum(new_meta)
        ewinrate = np.dot(winrate_mat, new_meta)
        # if ic % 1000 == 0:
        #     print(ic)
        #     print(new_meta)
        #     print(ewinrate)
        dif = ewinrate.max() - ewinrate[np.nonzero(new_meta)].min()
        ic += 1
        if ic >= 10000:
            print("max iter reached, no results found.")
            print(ewinrate)
            return new_meta, dif
    return new_meta, dif


if __name__ == "__main__":
    # load Meta data and Win rate data
    cur_meta = np.loadtxt('SampleMeta.csv')
    winrate_mat = np.loadtxt('SampleWinrate.csv', delimiter=',')

    # load Rock-Paper-Scissors model
    # winrate_mat = np.loadtxt('RPSWinrate.csv', delimiter=',')
    # cur_meta = np.random.random_sample([3])
    # cur_meta /= sum(cur_meta)

    # use random generator
    # np.random.seed(123)
    # N_CLASS = 6
    # winrate_mat = rdg.GenRandomWinrateMatrix(N_CLASS)
    # cur_meta = rdg.GenRandomMeta(N_CLASS)

    N_CLASS = int(cur_meta.shape[0])
    cur_meta = cur_meta.reshape([N_CLASS, 1])

    ewinrate = np.dot(winrate_mat, cur_meta)

    new_meta, dif = meta_evo(winrate_mat, cur_meta, 10 ** -2)
    print(new_meta)