#!/usr/bin/env python3

from itertools import combinations, product
from numpy.random import random
from numpy import percentile
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse

sns.set(font="monospace")


def scale(v, center=False):
    '''
    Scale the variables to a <0,1> range.
    If center == True, also centering on 0 is performed.
    '''
    v = (v - min(v))/(max(v)-min(v))
    if center:
        v = v - (sum(v)/len(v))
    return v


def score(v0, v1):
    '''
    Returns a measure of dissimilarity between two vectors.
    Here, the measure is mean squared differences.
    '''
    assert len(v0) == len(v1)
    msd = sum([(i-j)**2 for i,j in zip(v0, v1)]) / len(v0)
    return msd


def results(d, permute=False, pairs='all'):
    res = {}
    if pairs == 'all':
        pairs = combinations(d.columns, 2)
    if pairs == 'left-right':
        pairs = [(i,j) for i,j in combinations(d.columns, 2) if i[3:] == j[3:]]
    i=0
    for p in pairs:
        i+=1
        v0 = d[p[0]]
        v1 = d[p[1]]

        if permute:
            sum_sq_er = score(v0, v1.sample(len(v1)))
        else:
            sum_sq_er = score(v0, v1)

        res[p] = sum_sq_er
    o = pd.DataFrame.from_dict([i for i in zip(res.keys(), res.values())])
    o.columns = ['pair', 'score']
    return o.sort_values('score')
    return res


def get_threshold_val(res_rand, alpha):
    return percentile( res_rand['score'] , alpha*100)


def get_least_linear(results, results_random, alpha):
    thr = get_threshold_val(results_random, alpha)
    return results['pair'].ix[results['score'] > thr]


def get_most_linear(results, results_random, alpha=None, n=None):
    if alpha:
        thr = get_threshold_val(results_random, alpha)
        return results['pair'].ix[results['score'] < thr].tolist()
    if n:
        return results['pair'][:n].tolist()


def plot_relationships_ipynb(nl, d, n=3):
    if n > 20:
        print("Watch out, many plots will be generated")
    for i,j in nl[:n]:
        sns.lmplot(i, j, data=d, size=3)
        plt.axis([-0.1, 1.1, -0.1, 1.1])


def plot_relationships_cl(nl, d, plot_filename, n=3):
    counter = 0
    if n > 20:
        print("Watch out, many plots will be generated")
    for i,j in nl[:n]:
        plt.figure(3)
        sns.lmplot(i, j, data=d, size=3)
        plt.axis([-0.1, 1.1, -0.1, 1.1])
        plt.savefig(plot_filename + str(counter) + ".png")
        plt.close()
        counter +=1


def get_rand_data(nrows, ncols):
    d_rand = pd.DataFrame([random(ncols) for i in range(nrows)])
    d_rand = d_rand.apply(scale)
    return d_rand


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pairwise comparison of\
                        variables.")
    parser.add_argument("-i", "--input_file",
                        help="Path of the data frame which columns are to be\
                        compared.",
                        required=True)
    parser.add_argument("-o", "--output_dir",
                        help="Where the plots shall be saved.",
                        required=True)
    parser.add_argument("-t", "--threshold",
                    help="Percentile below which a relationship is\
                        considered to be linear. 1-threshold is used\
                        as percentile above which the relationship is\
                        considered to be non-linear\
                        ",
                        default=0.1,
                        required=False)
    args = vars(parser.parse_args())

    ifile_path = args['input_file']
    odir_path  = args['output_dir']
    alpha      = args['threshold']

    if not odir_path[-1] == "/":
        odir_path = odir_path+"/"

    p_fp1 = odir_path + "distplot_random"
    p_fp2 = odir_path + "distplot_data"
    p_fp3 = odir_path + "lmplot_least_linear"
    p_fp4 = odir_path + "lmplot_most_linear"
    ml_fp = odir_path + "most_linear.txt"
    nl_fp = odir_path + "least_linear.txt"
    n_rand = 1000 # number of permutations
    n_plots = 20 # max n plots to draw

    if not os.path.isdir(odir_path):
        os.mkdir(odir_path)

    for fp in [ml_fp, nl_fp]:
        if not os.path.isfile(fp):
            os.mknod(fp)

    d = pd.read_csv(ifile_path, sep='\t')
    d = d.apply(scale)

    d_rand = get_rand_data(d.shape[0], n_rand)

    res_rand = results(d_rand)
    res_acc = results(d)
    nl = get_least_linear(res_acc, res_rand, alpha = 1-alpha)
    ml = get_most_linear(res_acc, res_rand, alpha=alpha) #, n=alpha)

    plt.figure(1)
    sns.distplot([i for i in res_rand['score']])
    plt.savefig(p_fp1 + ".png")
    plt.close()

    plt.figure(2)
    sns.distplot([i for i in res_acc['score']]);
    plt.savefig(p_fp2 + ".png")
    plt.close()

    with open(ml_fp, 'w') as f:
        for relationship in ml:
            f.write(" - ".join(relationship)+'\n')

    with open(nl_fp, 'w') as f:
        for relationship in nl:
            f.write(" - ".join(relationship)+'\n')

    plot_relationships_cl(nl, d, p_fp3, n_plots)
    plot_relationships_cl(ml, d, p_fp4, n_plots)


