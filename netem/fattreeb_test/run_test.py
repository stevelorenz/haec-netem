#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# Author : Xiang, Zuo
# Email  : xianglinks@gmail.com
# About  : Run tests defined in testbench.py
"""

import sys
from os import makedirs, path
from time import strftime

import testbench
from MaxiNet.Frontend import maxinet
from topo import FatTreeBinary


def main():
    """Main function

    For each test function, two important objects:
        1. MaxiNet.Cluster: cluster of workers
        2. Mininet.Topo: to be tested topology
    """
    # setup MaxiNet cluster
    # without arguments, returns all the registered Workers
    cluster = maxinet.Cluster()

    # create with timestamp specified result data path
    sp_path = path.realpath(sys.argv[0])
    sp_dir = path.dirname(sp_path)
    # use current date to create unique path
    data_path = path.join(sp_dir,
                          'test_result_%s' % strftime('%Y%m%d_%H%M'))

    if not path.isdir(data_path):
        makedirs(data_path)

    # create topology and run tests
    # -----------------------------------------------------
    # -- partition using METIE --
    topo = FatTreeBinary(8, 2, 10)

    testbench.latency_test(cluster, topo, 5, data_path)
    testbench.loss_test(cluster, topo, (5, 5), data_path)
    testbench.bw_test_symmetric(cluster, topo, (5, 5), data_path)

    # -- TODO using static mapping --
    # -----------------------------------------------------

if __name__ == "__main__":
    main()
