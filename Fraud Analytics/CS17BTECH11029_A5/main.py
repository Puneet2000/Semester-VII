#!/usr/bin/env python3
import os
import copy
import math
import time
import argparse
import numpy as np
import pandas as pd
import pickle as pkl
import os.path as  osp

from collections import defaultdict
from collections import Counter

__AUTHORS___ = [
    ('YASH KHASBAGE', 'CS17BTECH11044'),
    ('PUNEET MANGLA', 'CS17BTECH11029'),
    ('RUSHIKESH TAMMEWAR', 'CS17BTECH11041')
]

"""
To run: python3 main.py
"""
def get_parser():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # parser
    # parameters needed for the algorithm
    parser.add_argument('--k', type=int, default=4, help='k nearest neighbours')
    parser.add_argument('--m', type=int, default=1, help='m parameter')
    parser.add_argument('--h', type=float, default=0.6, help='h parameter')
    parser.add_argument('-pdb', action='store_true', help='run with pdb')

    return parser

# stock flow graph
class Graph:
    def __init__(self, tids):
        # trader ids are stored as vertices
        self.vertices = set(tids)
        # adjacency dictionary stores every transaction as dictionary
        self.adj = defaultdict(dict)
        # this will be reverse adjacency dictionary
        # for example, u -> v transaction will be stored as v -> u
        # used for reducing time complexity
        self.revadj = None

    # adds edge to graph
    def add_edge(self, sid, bid, amt):
        # do not consider any trader that is present in
        # only one of seller or buyer
        if sid not in self.vertices and bid not in self.vertices:
            return
        # for repeated trade
        if bid in self.adj[sid]:
            self.adj[sid][bid] += amt
        else: # for a new trade
            self.adj[sid][bid] = amt

    # getter for vertices
    def get_vertices(self):
        return self.vertices

    # consider only k-nearest vertices
    # vertices with heighest weight are consider as nearest
    def maxk_neighbours(self, k):
        for u in self.vertices:
            knn = [(v, w) for (v, w) in self.adj[u].items()]
            knn = sorted(knn, key=lambda x: x[1], reverse=True)[:k]
            self.adj[u] = {v: w for (v, w) in knn}

    def store_reverse_edges(self):
        # reverse the edges
        # only for reducing computational complexity
        assert self.revadj is None, 'this function was called earlier too. not allowed'
        self.revadj = defaultdict(dict)
        for u in self.adj:
            for v in self.adj[u]:
                self.revadj[v][u] = self.adj[u][v]

    def summary(self):
        # simple summary of graph
        def print_char_line(c):
            print(c * 50)
        print_char_line('-')
        print('printing graph summary')
        print('# of vertices', len(self.vertices))
        print('few adjacency lists')
        # print some vertices with neighbours
        for i, u in enumerate(self.adj):
            if i == 5:
                break
            print('vertex', u, 'neighbours', self.adj[u])
        print('max neighbours', max([len(l) for l in self.adj.values()]))
        print('min neighbours', min([len(l) for l in self.adj.values()]))

        print_char_line('-')

# collusion clustering functions and algorithm
class CollusionClustering:
    def __init__(self, graph, k, m, h):
        # stock flow graph
        self.graph = graph
        # parameters of algorithm
        self.k = k
        self.m = m
        self.h = h
        # some restrictions mentioned by paper on
        # the parameters
        assert 1 <= self.m <= self.k
        assert 0 <= self.h <= 1

    def get_sorted_set_pairs(self, S):
        # create a list of all possible pair of clusters
        print('sorting pairs...')
        allpairs = list()
        nsets = len(S)
        # iterate over all clusters
        for i in range(nsets):
            for j in range(i+1, nsets):
                c1 = S[i]
                c2 = S[j]
                # compute collusion level
                collusion_level = self.collusion_level(c1, c2)
                allpairs.append((c1, c2, collusion_level))
        # sort by decreasing collusion level
        allpairs.sort(key=lambda x: x[2], reverse=True)
        return allpairs

    def get_collusion_index(self, C):
        # singleton set that does not trade with itself
        # will get index 0
        if len(C) == 1:
            c = next(iter(C))
            if c not in self.adj[c]:
                return 0

        internal_trade = 0
        external_trade = 0

        adj = self.graph.adj
        revadj = self.graph.revadj

        for sid in C:
            # iterate over transaction where seller is sid
            for bid in adj[sid]:
                if bid in C:
                    internal_trade += adj[sid][bid]
                else:
                    external_trade += adj[sid][bid]
            # iterate over transaction where sid is buyer
            for bid in revadj[sid]:
                if bid not in C:
                    external_trade += revadj[sid][bid]

        # when external trade is 0, we return index as a big number
        if external_trade == 0:
            return 10 ** 4

        return internal_trade / external_trade

    def collusion_level(self, set1, set2):
        # collusion level is defined only for disjoint sets
        assert set1.isdisjoint(set2)
        # get collusion index of union of sets
        return self.get_collusion_index(set1 | set2)

    def km_compatibility(self, p, C):
        # see intersection of knn of p and C
        knn_p = self.graph.adj[p]
        if len(set(knn_p) & C) >= min(self.m, len(C)):
            return True
        return False

    def khm_compatibility(self, C, D):
        # check compatibility from C side
        c_compatible = 0
        for c in C:
            if self.km_compatibility(c, D):
                c_compatible += 1
        # check compatibility from D side
        d_compatible = 0
        for d in D:
            if self.km_compatibility(d, C):
                d_compatible += 1
        # check if sets are compatible from both C and D side
        if c_compatible >= self.h * len(C) and d_compatible >= self.h * len(D):
            return True
        return False

    def cluster(self):
        # cluster the vertices
        vertices = self.graph.get_vertices()

        # create set of all singleton clusters
        S = list({u} for u in vertices)

        # get all cluster pairs arranged according to collusion index
        B = self.get_sorted_set_pairs(S)

        while True:
            print('total trader sets:', len(S))
            merged_sets = None
            print('iterating over all pairs...')
            # find the set pair having khm compatibility
            # and also highest collusion index
            for C, D, Lc in B:
                if Lc > 0 and self.khm_compatibility(C, D):
                    merged_sets = (C, D)
                    # remove one set and add it to other set
                    C.update(D)
                    S.remove(D)
                    break
            if merged_sets is None:
                # if no merging possible, then stop the algorithm
                break

            new_B = list()
            for i in range(len(B)):
                # collection all cluster pairs that do not involve
                # the merged sets
                collusion_pair = B[i]
                if collusion_pair[0] in merged_sets:
                    continue
                if collusion_pair[1] in merged_sets:
                    continue
                # store all such paris
                new_B.append(collusion_pair)
            merged_set = merged_sets[0] | merged_sets[1]
            # compute the merged set
            replacement_pairs = list()
            for i in range(len(S)):
                # compute the possible new cluster pairs that
                # are formed due to the merged cluster
                c1 = S[i]
                c2 = merged_set
                if c1 == c2:
                    continue
                # compute all collusion levels and add pairs to list
                collusion_level = self.collusion_level(c1, c2)
                replacement_pairs.append((c1, c2, collusion_level))

            new_B = new_B + replacement_pairs
            # again sort the cluster pairs with decreasing
            # collusion index
            new_B.sort(key=lambda x: x[2], reverse=True)
            B = new_B
        return S


if __name__ == '__main__':

    args = get_parser().parse_args()

    # k = 4
    # m = 1
    # h = 0.6
    k = args.k
    m = args.m
    h = args.h

    if args.pdb:
        import pdb
        pdb.set_trace()

    # dataset path
    dataset_path = 'dataset.csv'
    # read dataset
    df = pd.read_csv(dataset_path, header=None, names=['sid', 'bid', 'amt'], dtype={'sid': int, 'bid': int, 'amt': int})

    unique_sid = df['sid'].unique()
    unique_bid = df['bid'].unique()
    # unique sellers and buyers
    print('unique_sid', len(unique_sid))
    print('unique_bid', len(unique_bid))
    tids = np.intersect1d(unique_sid, unique_bid)
    # find traders that can atleast form a cycle
    # these are traders that are present both as sellers and buyers
    print('# of traders candidate for circular trading:', len(tids))

    # stock flow graph
    print('some example neighbours')
    sfg = Graph(tids)
    # add edges to graph
    for idx, row in df.iterrows():
        sid = row[0]
        bid = row[1]
        amt = row[2]

        sfg.add_edge(sid, bid, amt)

    print('reducing nearest neighbours')
    # compute knn
    sfg.maxk_neighbours(k)
    print('storing reverse edges')
    # store reverse edges
    sfg.store_reverse_edges()

    print('summary of graph')
    sfg.summary()

    print('stock flow graph complete')
    print('creating CollusionClustering object')
    # create instane of algo runner
    collusion_clustering = CollusionClustering(sfg, k, m, h)

    print('clustering begins...')
    # cluster
    clusters = collusion_clustering.cluster()

    print('cluster size: # of clusters with that size')
    # print frequency of cluster sizes
    print(Counter([len(c) for c in clusters]))

    print('clusters:')
    # print non-trivial clusters
    for c in clusters:
        if len(c) < 2:
            continue
        print(c, end=' ')

    print()
    print('only big clusters with collusion index')
    # print clusters and their collusion index
    for c in clusters:
        if len(c) >= 3:
            print('cluster', c, 'collusion index', collusion_clustering.get_collusion_index(c))
