#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
# Author : Xiang, Zuo
# Email  : xianglinks@gmail.com
# About  : Data processing and Plotting for binary fat tree test
           These codes are just personal use for creation figures, so is not so
           structured, just ignore this
"""

import sys
from os import path

import ipdb
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib import lines as mlines
from matplotlib import pyplot as plt


# === data proc functions ===

def get_node_dist(host_num):
    """Get node distance in binary fattree topo

    Args:
        host_num (int): number of host

    Returns:
        dist_lt (list): list of distance from h1 to other nodes

    """

    if (np.log2(host_num) - int(np.log2(host_num))) == 0:
        tree_depth = int(np.log2(host_num))
    else:
        tree_depth = int(np.log2(host_num) + 1)

    full_leaf_num = int(pow(2, tree_depth))  # number of leaves

    dist_lt = [0] * full_leaf_num
    double_tree_depth = 2 * tree_depth
    lca_dist = 0  # lowest common ancestor
    todo_lt = 0
    todo_num = int(full_leaf_num / 2)

    while todo_num >= 1:
        # calc current distance
        todo_lt = int(double_tree_depth - 2 * lca_dist)
        dist_lt[todo_num:2 * todo_num] \
            = [todo_lt] * int(todo_num)
        lca_dist += 1  # increase LCA
        todo_num = int(todo_num / 2)
    dist_lt = dist_lt[:host_num]

    return dist_lt


def get_th_lt(host_num, para_flag, para_value):
    """Get theoritical link parameters in a list

    Args:
        host_num (int): number of host
        para_flag (str): flag for different parameter
        para_value (float): values for link parameter

    Returns:
        th_lt (list): list of theoritical link parameter

    """
    dist_lt = get_node_dist(host_num)

    if para_flag == 'latency':
        th_lt = list(map(lambda x: 2 * x * para_value, dist_lt))
        return th_lt

    if para_flag == 'loss':
        th_lt = []
        for dist in dist_lt:
            # convert to procent
            th_lt.append(
                100.0 * (1 - np.power(float(1 - (para_value / 100.0)),
                                      2 * dist))
            )
        return th_lt


def calc_rel_diff(avg_df, std_df, para_flag):
    """Calc relative deviation

    Args:
        avg_df (Dataframe): Dataframe of average values
        std_df (Dataframe): Dataframe of standard deviation values
        para_flag (str): flag for different link parameter

    Returns:
        rel_diff_dic (dict): relative difference average and std values
    """
    host_lt = avg_df.columns.values.tolist()
    para_lt = avg_df.index.tolist()

    reldiff_avg_lt = []
    reldiff_std_lt = []

    # calc avg and std value for relative differece
    # for std is assumed that the rel_diff values are i.i.d

    for para in para_lt:
        tmp_sum = 0
        std_sum = 0
        # get theoritical parameter list
        th_lat_lt = get_th_lt(len(host_lt) + 1, para_flag, para)

        for host in host_lt:
            # clac average value
            tmp_sum += np.abs(avg_df.at[para, host] - th_lat_lt[host-1])\
                / th_lat_lt[host-1]
            # calc std deviation
            # assumed that values of different host are independent
            std_sum += (std_df.at[para, host] / np.sqrt(len(host_lt)))

        reldiff_avg_lt.append(float(tmp_sum) / len(host_lt))
        reldiff_std_lt.append(float(std_sum) / np.power(len(host_lt), 2))

    rel_diff_dic = {
        'avg': reldiff_avg_lt,
        'std': reldiff_std_lt
    }

    return rel_diff_dic


def calc_rdiff_bw(bw_data_df):
    """Calc relative different for bandwidth test

    Args:
        bw_data_df (Dataframe):

    Returns:
        rel_diff_dic (dict):

    """

    # drop columns of timestamp
    bw_lt = bw_data_df.index.get_level_values('bw').unique().tolist()
    host_lt = bw_data_df.index.get_level_values('end_host').unique().tolist()

    avg_lt = []
    std_lt = []

    # sum(abs(test - theoritical) / theoritical) / 32

    # get average value and st-deviation
    for bw in bw_lt:
        tmp_avg = 0
        tmp_std = 0
        for host in host_lt:
            tmp_avg = tmp_avg +\
                (np.abs(bw_data_df.loc[bw, host].mean()[0] - bw * np.power(10, 6))) /\
                (bw * np.power(10, 6))
            # assumed the distribution between host is independent
            tmp_std = tmp_std +\
                (bw_data_df.loc[bw, host].std()[0]) / np.sqrt(len(host_lt)) /\
                np.power((bw * np.power(10, 6)), 2)
        avg_lt.append(tmp_avg / len(host_lt))
        std_lt.append(tmp_std / np.power(len(host_lt), 2))

    # convert results to procent
    avg_lt = [x * 100 for x in avg_lt]
    std_lt = [x * 100 for x in std_lt]

    rel_diff_dic = {
        'avg': avg_lt,
        'std': std_lt
    }

    return rel_diff_dic


def calc_util(cpu_df, mem_df):
    """Calc average cpu and memory untilization

    Args:
        cpu_df (Dataframe): df for cpu log
        mem_df (Dataframe): df for mem log

    """

    cpu_lt = [ cpu_df.loc[bw].mean()['%sum'] for bw in range(1, 6) ]
    mem_lt = [
        100.0 * (1 - mem_df.loc[bw].mean()[0] / (2 * 1024 * 1024))
        for bw in range(1, 6)
    ]

    return cpu_lt, mem_lt


# === plot functions ===
def plot_reldiff_lat(lattest_df_lt):
    """Plot relative difference for latency test

    Args:
        lattest_df_lt (list): list of latency test results Dataframe

    """

    rel_diff_lt = []  # list of relative difference

    for lattest_df in lattest_df_lt:
        lat_stat_df = lattest_df[['link_delay', 'end_host', 'avg', 'std']]

        # dataframe for average and std values
        # sort by end_host index
        avg_df = lat_stat_df.pivot('link_delay',
                                   'end_host',
                                   'avg').sort_index(axis=1)
        std_df = lat_stat_df.pivot('link_delay',
                                   'end_host',
                                   'std').sort_index(axis=1)

        # calc relative difference
        rel_diff_lt.append(calc_rel_diff(avg_df, std_df, 'latency')['avg'])

    # save data for latex table
    delay_lt = range(1, len(rel_diff_lt[0]) + 1)
    # init relative difference DataFrame
    rel_diff_df = pd.DataFrame(
        rel_diff_lt, index=['wk_num 1', 'wk_num 16'], columns=list(delay_lt)
    )
    rel_diff_df = rel_diff_df.round(3)  # round with 3 decimal places

    with open('./lattest_reldiff_table.tex', 'w+') as table_file:
        rel_diff_df.to_latex(buf=table_file, longtable=None)

    # plot rel_diff_lt
    plt.figure(1)
    delay_lt = range(1, len(rel_diff_lt[0]) + 1)

    for index, _ in enumerate(rel_diff_lt):
        rel_diff_lt[index] = list(map(lambda x: x * 100, rel_diff_lt[index]))

    plt.plot(delay_lt, rel_diff_lt[0], color='black',
             label='wk_num= 1', lw=1, ls='-', marker='o', markevery=1,
             markersize=4, markerfacecolor='None',
             markeredgewidth=1, markeredgecolor='black')

    plt.plot(delay_lt, rel_diff_lt[1], color='blue',
             label='wk_num=16', lw=1, ls='-', marker='s',
             markevery=1, markersize=4, markerfacecolor='None',
             markeredgewidth=1, markeredgecolor='blue')

    # plot a upper-bounds
    upper_bound = [5] * len(rel_diff_lt[0])
    plt.plot(delay_lt, upper_bound, label='5% bound',
             ls='--', lw=1.0, color='red')

    upper_bound = [3] * len(rel_diff_lt[0])
    plt.plot(delay_lt, upper_bound, label='3% bound',
             ls='--', lw=1.0, color='green')

    plt.legend(loc='best', fontsize=11, handlelength=2.5)
    plt.xlim(1, 30)
    plt.xlabel('link delay (ms)')
    plt.ylabel('deviation value (%)')
    plt.grid()
    plt.savefig('./lattest_reldiff_plot.png', dpi=600, bbox_inches='tight')


def plot_lat_byhost(lattest_df):
    """Plot latency for different host with specific link delays

    Args:
        lattest_df (Dataframe): Dataframe of latency results
        delay_lt (list):

    """
    # calc avg and std, ignore the first ping value
    lat_stat_df = lattest_df[['link_delay', 'end_host', 'avg', 'std']]

    # dataframe for average and std values
    # sort by end_host index
    avg_df = lat_stat_df.pivot('link_delay',
                               'end_host',
                               'avg').sort_index(axis=1)
    std_df = lat_stat_df.pivot('link_delay',
                               'end_host',
                               'std').sort_index(axis=1)

    host_lt = avg_df.columns.values.tolist()

    # plot theoritical values
    plt.figure(2)
    x_axis = range(1, 33)
    host_num = len(host_lt) + 1
    th_lat_lt = get_th_lt(host_num, 'latency', 5)

    plt.plot(x_axis, th_lat_lt, lw=1.0, ls='None', marker='x', color='black',
             markersize=4, markeredgewidth=1.5, label='theoritical, delay=5ms')

    th_lat_lt = get_th_lt(host_num, 'latency', 15)

    plt.plot(x_axis, th_lat_lt, lw=1.0, ls='None', marker='x', color='blue',
             markersize=4, markeredgewidth=1.5,
             label='theoritical, delay=15ms')

    th_lat_lt = get_th_lt(host_num, 'latency', 25)

    plt.plot(x_axis, th_lat_lt, lw=1.0, ls='None', marker='x', color='green',
             markersize=4, markeredgewidth=1.5,
             label='theoritical, delay=25ms')

    # plot measured values with confidence intervall
    t_factor = 3.496  # for 99.9%

    test_lat_lt = avg_df.loc[5].tolist()
    test_lat_lt.insert(0, 0)
    yerror_lt = list(map(lambda x: x * t_factor / np.sqrt(49),
                         std_df.loc[5].tolist()))
    yerror_lt.insert(0, 0)
    plt.errorbar(x_axis, test_lat_lt, yerr=yerror_lt, lw=1.5,
                 ls='None', marker='None', markersize=4, color='black',
                 label='tested, delay=5ms')

    test_lat_lt = avg_df.loc[15].tolist()
    test_lat_lt.insert(0, 0)
    yerror_lt = list(map(lambda x: x * t_factor / np.sqrt(50),
                         std_df.loc[15].tolist()))
    yerror_lt.insert(0, 0)
    plt.errorbar(x_axis, test_lat_lt, yerr=yerror_lt, lw=1.5, ls='None',
                 marker='None', markersize=4, color='blue',
                 label='tested, delay=15ms')

    test_lat_lt = avg_df.loc[25].tolist()
    test_lat_lt.insert(0, 0)
    yerror_lt = list(map(lambda x: x * t_factor / np.sqrt(50),
                         std_df.loc[25].tolist()))
    yerror_lt.insert(0, 0)
    plt.errorbar(x_axis, test_lat_lt, yerr=yerror_lt, lw=1.5, ls='None',
                 marker='None', markersize=4, color='green',
                 label='tested, delay=25ms')

    plt.xlim(1, 33)
    # plt.ylim(0, 150)
    plt.legend(loc='best', fontsize=9)
    plt.ylabel('latency (ms)')
    plt.xlabel('host index')

    plt.grid()
    plt.savefig('./lattest_latbyhost_plot.png', dpi=600, bbox_inches='tight')


def plot_reldiff_bw(bw_df_lt, cpu_df, mem_df):
    """Plot relative difference bandwidth

    Args:
        bw_df_lt (list):
    """

    # get relative different dict
    rel_diff_lt = [calc_rdiff_bw(x) for x in bw_df_lt]

    # plot reldiff with two scales
    # use two different axes
    plt.figure(4)

    # first axis, ax1
    # -------------------------------------------
    ax1 = plt.axes()  # get axis of figure

    ind = np.arange(len(rel_diff_lt[0]['avg']))
    width = 0.35

    # add bars
    bar1 = ax1.bar(ind, rel_diff_lt[0]['avg'], width, color='b',
                   alpha=0.6, edgecolor='black', lw=1,
                   label='deviation wk_num=1')

    bar2 = ax1.bar(ind + width, rel_diff_lt[1]['avg'], width, color='g',
                   alpha=0.6, edgecolor='black',lw=1,
                   label='deviation wk_num=16')

    ax1.set_xlabel('host bandwidth (Mbps)')
    x_labels = bw_df_lt[0].index.get_level_values('bw').unique().tolist()
    plt.xticks(ind + width, x_labels)
    ax1.set_ylabel('deviation value (%)')

    cpu_util, mem_util = calc_util(cpu_df, mem_df)

    ax2 = ax1.twinx()

    ln2 = ax2.plot(ind + width * 3.0 / 2, cpu_util, color='red', lw=1.2,
             marker='o', markerfacecolor='None', markeredgewidth=1.2,
                markeredgecolor='red', label='cpu untilization wk_num=16')

    # get legends of two axis
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='best', fontsize=9, handlelength=2.5)

    ax2.set_ylabel('CPU untilization (%)')

    # both configs
    plt.savefig('./bwtest_reldiff_plot.png', dpi=600, bbox_inches='tight')


def plot_boxplot_bw(bw_df_lt):
    """Plot Boxplot for distribution of bandwidth tests

    Args:
        bw_df_lt (list): list of bandwidth test dataframes
    """
    # reindex
    reindex_df_lt = [x.reset_index(level=1)['bits_per_second']
                     for x in bw_df_lt]

    mean_line = mlines.Line2D([], [], color='g', label='mean')
    median_line = mlines.Line2D([], [], color='r', label='median')

    # -- worker_num 1 --
    box_data = []
    for bw in range(1, 6):
        # convert to Mbps
        box_data.append(
            list(map(lambda x: float(x) / np.power(10, 6),
                     reindex_df_lt[0].loc[bw].tolist()))
        )

    plt.figure(5)
    box1 = plt.boxplot(box_data, vert=True, sym='r+',
                       showmeans=1, meanline=1)
    # set plot parameters
    plt.setp(box1['means'], color='g')

    ax = plt.axes()
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, which='major', alpha=0.5)
    plt.xlabel('host bandwidth (Mbps)')
    plt.ylabel('tested bandwidth (Mbps)')
    plt.legend(handles=[mean_line, median_line],
               loc='upper left', fontsize=12)
    plt.ylim(0.5, 5.0)

    plt.savefig('./bwtest_boxplot_wn1.png', dpi=600, bbox_inches='tight')

    # -- worker_num 16 --
    box_data = []
    for bw in range(1, 6):
        box_data.append(
            list(map(lambda x: float(x) / np.power(10, 6),
                     reindex_df_lt[1].loc[bw].tolist()))
        )

    plt.figure(6)
    box2 = plt.boxplot(box_data, vert=True, sym='r+',
                       showmeans=1, meanline=1)
    # set plot parameters
    plt.setp(box2['means'], color='g')

    ax = plt.axes()
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, which='major', alpha=0.5)
    plt.xlabel('host bandwidth (Mbps)')
    plt.ylabel('tested bandwidth (Mbps)')
    plt.legend(handles=[mean_line, median_line],
               loc='upper left', fontsize=12)
    plt.ylim(0.5, 5.0)

    plt.savefig('./bwtest_boxplot_wn16.png', dpi=600, bbox_inches='tight')


def plot_reldiff_loss(loss_df):
    """Plot relative difference of loss rate

    Args:
        loss_df (Dataframe)
    """
    loss_stat_df = loss_df[['link_loss', 'end_host', 'avg', 'std']]

    # dataframe for average and std values
    # sort by end_host index
    avg_df = loss_stat_df.pivot('link_loss',
                                'end_host',
                                'avg').sort_index(axis=1)
    std_df = loss_stat_df.pivot('link_loss',
                                'end_host',
                                'std').sort_index(axis=1)

    rel_diff = calc_rel_diff(avg_df, std_df, 'loss')['avg']
    # std_lt = calc_rel_diff(avg_df, std_df, 'loss')['std']

    rel_diff = list(map(lambda x: x * 100.0,
                        rel_diff))

    # plot using bar plot
    # plot the results
    plt.figure(6)
    ax = plt.axes()  # get axis of figure

    plt.plot(range(1, 11), rel_diff, color='black', ls='-', marker='o',
             lw=1.5, markerfacecolor='None', markeredgewidth=1.5,
             markeredgecolor='black', label='wk_num=16')

    upper_bound = [3] * 2 * len(rel_diff)
    plt.plot(range(2 * len(rel_diff)), upper_bound, label='3% bound',
             ls='--', lw=1.5, color='blue')

    # picture configs
    plt.legend(loc='best', fontsize=11, handlelength=2.5)
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)
    plt.xlim(1, 10)
    plt.xlabel('link loss rate (%)')
    plt.ylabel('deviation value (%)')
    plt.savefig('./losstest_reldiff_plot.png', dpi=600, bbox_inches='tight')


def plot_loss_byhost(loss_df):
    """Plot loss rate for different host with specific link loss

    Args:
        loss_df (Dataframe): Dataframe of latency results
    """
    # calc avg and std, ignore the first ping value
    loss_stat_df = loss_df[['link_loss', 'end_host', 'avg', 'std']]

    # dataframe for average and std values
    # sort by end_host index
    avg_df = loss_stat_df.pivot('link_loss',
                                'end_host',
                                'avg').sort_index(axis=1)
    std_df = loss_stat_df.pivot('link_loss',
                                'end_host',
                                'std').sort_index(axis=1)

    host_lt = avg_df.columns.values.tolist()

    # plot theoritical values
    plt.figure(7)
    x_axis = range(1, 33)
    host_num = len(host_lt) + 1
    th_lat_lt = get_th_lt(host_num, 'loss', 2)
    plt.plot(x_axis, th_lat_lt, lw=1.0, ls='None', marker='x', color='black',
             markersize=4, markeredgewidth=1.5, label='theoritical, loss=2%')

    th_lat_lt = get_th_lt(host_num, 'loss', 5)
    plt.plot(x_axis, th_lat_lt, lw=1.0, ls='None', marker='x', color='blue',
             markersize=4, markeredgewidth=1.5, label='theoritical, loss=5%')

    th_lat_lt = get_th_lt(host_num, 'loss', 8)
    plt.plot(x_axis, th_lat_lt, lw=1.0, ls='None', marker='x', color='green',
             markersize=4, markeredgewidth=1.5, label='theoritical, loss=8%')

    # plot measured values with confidence intervall
    t_factor = 3.496  # for 99.9%
    test_lat_lt = avg_df.loc[2].tolist()
    test_lat_lt.insert(0, 0)
    yerror_lt = list(map(lambda x: x * t_factor / np.sqrt(49),
                         std_df.loc[2].tolist()))
    yerror_lt.insert(0, 0)
    plt.errorbar(x_axis, test_lat_lt, yerr=yerror_lt, lw=1.5, ls='None',
                 marker='None', markersize=4, color='black',
                 label='tested, loss=2%')

    test_lat_lt = avg_df.loc[5].tolist()
    test_lat_lt.insert(0, 0)
    yerror_lt = list(map(lambda x: x * t_factor / np.sqrt(49),
                         std_df.loc[5].tolist()))
    yerror_lt.insert(0, 0)
    plt.errorbar(x_axis, test_lat_lt, yerr=yerror_lt, lw=1.5, ls='None',
                 marker='None', markersize=4, color='blue',
                 label='tested, loss=5%')

    test_lat_lt = avg_df.loc[8].tolist()
    test_lat_lt.insert(0, 0)
    yerror_lt = list(map(lambda x: x * t_factor / np.sqrt(49),
                         std_df.loc[8].tolist()))
    yerror_lt.insert(0, 0)
    plt.errorbar(x_axis, test_lat_lt, yerr=yerror_lt, lw=1.5, ls='None',
                 marker='None', markersize=4, color='green',
                 label='tested, loss=8%')

    plt.xlim(1, 33)
    # plt.ylim(0, 150)
    plt.legend(loc='best', fontsize=11)
    plt.ylabel('loss rate (%)')
    plt.xlabel('host index')

    plt.grid()
    plt.savefig('./losstest_latbyhost_plot.png', dpi=600, bbox_inches='tight')


# === main function ===
def main():
    """ main function """

    # get raw data and pass to proc functions
    script_path = path.realpath(sys.argv[0])
    script_dir = path.dirname(script_path)
    csv_data_dir = path.join(script_dir, '../../data_csv/fattreeb_test/')

    # --- latency test processing ---
    # -----------------------------------------------------
    lattest_df_lt = []  # lat test frame list
    for worker_num in [1, 16]:
        lattest_df_lt.append(pd.read_csv(
            path.join(csv_data_dir, 'lattest_hn32_wn%d.csv' % worker_num),
            sep=','
        ))

    # append avg and std to lattest_df
    for lattest_df in lattest_df_lt:
        # calc avg and std, ignore the first ping value
        lattest_df['avg'] = lattest_df.iloc[:, 3:].mean(1)
        lattest_df['std'] = lattest_df.iloc[:, 3:].std(1)

    # plot_reldiff_lat(lattest_df_lt)  # plot relative difference

    # plot_lat_byhost(lattest_df_lt[1])  # plot latency for different host


    # --- loss test processing ---
    # -----------------------------------------------------
    losstest_df = pd.read_csv(
        path.join(csv_data_dir, 'losstest_hn32_wn16.csv'),
        sep=','
    )

    losstest_df['avg'] = losstest_df.iloc[:, 3:].mean(1)
    losstest_df['std'] = losstest_df.iloc[:, 3:].std(1)

    # plot_reldiff_loss(losstest_df)
    # plot_loss_byhost(losstest_df)

    # -----------------------------------------------------

    # --- bandwidth test processing ---
    # -----------------------------------------------------
    bwtest_df_lt = []
    cpulog_df_lt = []
    for worker_num in [1, 16]:
        # - read bw test data -
        bwtest_df = pd.read_csv(
            path.join(csv_data_dir, 'bwtest_hn32_wn%d.csv' % worker_num),
            sep=',', index_col=['bw', 'end_host'],
        )[['bits_per_second']]

        # sort dataframe before multi-index
        # avoid PerformanceWarning, using sorted df is quicker
        bwtest_df = bwtest_df.sort_index(axis=0)

        # choose first 50 test results
        bwtest_df = pd.concat([bwtest_df.loc[bw, end_host].iloc[0:50]
                               for bw in range(1, 6)
                               for end_host in range(1,33)])

        bwtest_df_lt.append(bwtest_df)

    # - read log data -
    cpulog_df = pd.read_csv(
        path.join(csv_data_dir, 'cpu_bwtest_s31_hn32_wn16.log'),
        delim_whitespace=True,
        index_col = ['%bw']
    )[['%usr', '%sys']]
    cpulog_df['%sum'] = cpulog_df['%usr'] + cpulog_df['%sys']

    memlog_df = pd.read_csv(
        path.join(csv_data_dir, 'mem_bwtest_s31_hn32_wn16.log'),
        sep=',', index_col=['bw']
    )[['free']]

    plot_reldiff_bw(bwtest_df_lt, cpulog_df, memlog_df)  # plot relative difference
    plot_boxplot_bw(bwtest_df_lt)  # plot boxplot for bandwidth
    # -----------------------------------------------------


if __name__ == "__main__":
    main()
