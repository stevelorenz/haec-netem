#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# Author : Xiang, Zuo
# Email  : xianglinks@gmail.com
# About  : Test bench for binary fat tree topology
"""

import csv
import re
from os import path
from time import sleep

from logger import logger
from MaxiNet.Frontend import maxinet


# === Tool Functions ===
def sv_dist_node(exp, exp_flag, data_path):
    """Save node distribution of current experiment as CSV file

    Args:
        exp (MaxiNet.Experiment): to be saved experiment
        exp_flag (str): flag of current experiment
        data_path (str): path to save csv data
    """
    # set distribution file name
    dist_file_name = 'dist_%s_hn%d_wn%d.csv'\
        % (exp_flag, len(exp.origtopology.hosts()),
           exp.cluster.num_workers())

    dist_lt = []
    # exp.origtopology: unpartitioned topology of current exp
    for node in exp.origtopology.node_names():
        dist_lt.append([node, str(exp.get_worker(node).hn()),
                        str(exp.get_worker(node).ip())])

    # write dist in csv file
    with open(path.join(data_path, dist_file_name), 'w+') as dist_file:
        csv_writer = csv.writer(dist_file, delimiter=',')
        csv_writer.writerows(dist_lt)


def log_node_load(exp, exp_flag, node, log_path):
    """Log CPU and Mem usage of worker, on which node is
       emulated

    Use MaxiNet build-in log function
        CPU: mpstat, every 1 s
        MEM: /proc/meminfo, MemFree, Buffers, Cached

    Args:
        exp (MaxiNet.Experiment): to be saved experiment
        exp_flag (str): flag of current experiment
        node (str): node, whose worker will be logged
        log_path (str): path to save log data
    """
    # get to be logged worker
    worker = exp.get_worker(node)

    # CPU usage log
    # using mpstat, every second
    cpu_f_path = path.join(log_path, 'cpu_%s_%s_hn%d_wn%d.log'
                           % (exp_flag, node,
                              len(exp.origtopology.hosts()),
                              exp.cluster.num_workers()))
    worker.daemonize('mpstat 1 > %s' % cpu_f_path)

    # Memory usage log
    # using free command, every second
    mem_f_path = path.join(log_path, 'mem_%s_%s_hn%d_wn%d.log'
                           % (exp_flag, node,
                              len(exp.origtopology.hosts()),
                              exp.cluster.num_workers()))

    worker.daemonize_script(
        "getMemoryUsage.sh", " > " + mem_f_path
    )


# === Test Functions ===

def bw_test_symmetric(cluster, topo, test_para, data_path):
    """Symmetrical TCP bandwidth test for full binary tree

    This test is under condition of a full binary tree, which allows
    systematically partition into senders and receivers
    Using Iperf TCP mode, default window size 64 KB with single connection

    Args:
        cluster (maxinet.Cluster): cluster object
        topo (mininet.Topo): network topology
        test_para (tuple): (test rounds, test time)
            test time: -t option of Iperf, time of each test
            test rounds: number of test rounds for statistics
        data_path (str): path for saving csv results

    Raises:
        ValueError: if tree is not symmetric
    """
    host_list = topo.hosts()
    host_num = len(host_list)
    host_bw = topo.get_paradict()['host_bw']

    # choose senders and receivers systematically
    s_host_list = host_list[:int(host_num * 0.5)]
    r_host_list = host_list[int(host_num * 0.5):]

    if len(s_host_list) != len(r_host_list):
        raise ValueError('the length of sender list and \
                         receiver list should be same')

    exp = maxinet.Experiment(cluster, topo)
    exp.setup()

    logger.info('bandwidth test symmetric with host_bw: %d Mbps is setup',
                host_bw)

    # save node distribution
    sv_dist_node(exp, 'bwtest', data_path)

    # compile ping result re pattern
    pattern = re.compile(r"time=([\d]*[.]*[\d]*)")

    # === run bandwidth test ===
    # -- ping all round
    logger.info('start ping all round')
    for s_host in host_list:
        # loop for receive host
        for r_host in host_list:
            # ping until first success for warming up
            ping_cmd = 'ping -c 1 %s' % exp.get_node(r_host).IP()
            while True:
                result = exp.get_node(s_host).cmd(ping_cmd)
                if pattern.search(result):
                    break
        logger.debug('--- end ping all from %s', s_host)
    logger.info('finish ping all round')

    # -- TCP bandwidth test
    # -------------------------------------------
    logger.info('start TCP bandwidth test')

    # create data_path on each remote host
    # in order to run Iperf at background and write data in CSV file
    # here use get_node().cmd('xxx &'), which do not return result to Frontend
    # so need to create data file path distributed on each worker
    csv_file_path = path.join(data_path, 'bwtest_hn%d_wn%d'
                              % (host_num, exp.cluster.num_workers()))

    for node in topo.node_names():
        exp.get_node(node).cmd('mkdir -p %s' % data_path)

    # run load logger at background for root switch
    root_switch = topo.switch_names()[-1]
    log_node_load(exp, 'bwtest%d' % host_bw,
                  root_switch, data_path)

    # loop for test rounds
    for test_round in range(test_para[0]):
        # senders as client, recievers as server
        for host_index, r_host in enumerate(r_host_list):
            # the last host do not run Iperf in background
            if host_index == (len(r_host_list) - 1):
                ipf_client_cmd = 'iperf -c %s -t %d'\
                    % (exp.get_node(r_host).IP(), test_para[1])
            else:
                ipf_client_cmd = 'iperf -c %s -t %d &'\
                    % (exp.get_node(r_host).IP(), test_para[1])

            ipf_server_cmd = 'echo %d,%s,$(iperf -s -t %d -y C) >> \
                %s_$(hostname).csv &' % (host_bw, r_host,
                                         test_para[1], csv_file_path)

            exp.get_node(r_host).cmd(ipf_server_cmd)
            exp.get_node(s_host_list[host_index]).cmd(ipf_client_cmd)

        sleep(test_para[1] + 5)  # wait for Iperf server to save data

        # stop Iperf server on receivers
        for r_host in r_host_list:
            # return 0 if no process exist
            while int(exp.get_node(r_host).cmd('pgrep -c iperf')):
                exp.get_node(r_host).cmd('killall iperf')

        # inverse test s_host <-> r_host exchange
        s_host_list, r_host_list = r_host_list, s_host_list

        for host_index, r_host in enumerate(r_host_list):
            # the last host do not run iperf in background
            if host_index == (len(r_host_list) - 1):
                ipf_client_cmd = 'iperf -c %s -t %d'\
                    % (exp.get_node(r_host).IP(), test_para[1])
            else:
                ipf_client_cmd = 'iperf -c %s -t %d &'\
                    % (exp.get_node(r_host).IP(), test_para[1])

            ipf_server_cmd = 'echo %d,%s,$(iperf -s -t %d -y C) >> \
                %s_$(hostname).csv &' % (host_bw, r_host,
                                         test_para[1], csv_file_path)

            exp.get_node(r_host).cmd(ipf_server_cmd)
            exp.get_node(s_host_list[host_index]).cmd(ipf_client_cmd)

        sleep(test_para[1] + 5)

        # stop iperf server on receivers
        for r_host in r_host_list:
            while int(exp.get_node(r_host).cmd('pgrep -c iperf')):
                exp.get_node(r_host).cmd('killall iperf')

        logger.debug('--- end bw test of round %d', (test_round + 1))
    logger.info('bandwidth test finished')
    # -------------------------------------------

    logger.info('wait 2 seconds and stop the experiment')
    sleep(2)
    logger.info('stop current experiment')
    exp.stop()


def latency_test(cluster, topo, test_rounds, data_path):
    """Latency test between h1 and other hosts

    Use ping command, each round only ping once

    Args:
        cluster (maxinet.Cluster): cluster of workers
        topo (mininet.Topo): network topology
        test_rounds (int): number of ping rounds for statistics
        data_path (str): path for saving csv results
    """
    host_list = topo.hosts()
    host_num = len(host_list)
    link_delay = topo.get_paradict()['link_delay']

    exp = maxinet.Experiment(cluster, topo)
    exp.setup()

    # save node distribution
    sv_dist_node(exp, 'lattest', data_path)

    # print experiment info
    logger.info('latency test with link_delay: %d ms is setup', link_delay)

    # compile ping result re pattern
    pattern = re.compile(r"time=([\d]*[.]*[\d]*)")

    # === run latency test ===
    # -- ping all round
    logger.info('start ping all round')
    for s_host in host_list:
        # loop for receive host
        for r_host in host_list:
            # ping until first success for warming up
            ping_cmd = 'ping -c 1 %s' % exp.get_node(r_host).IP()
            while True:
                result = exp.get_node(s_host).cmd(ping_cmd)
                if pattern.search(result):
                    break
        logger.debug('--- end ping all of %s', s_host)
    logger.info('finish ping all round')

    # --- latency tests from host1
    # -------------------------------------------
    # csv data file
    csv_file_name = 'lattest_hn%d_wn%d.csv' % (host_num, cluster.num_workers())
    csv_file_path = path.join(data_path, csv_file_name)
    host_list.remove('h1')  # remove h1 from list
    logger.info('start latency test from h1')

    # run ping and process the result with matching pattern
    for r_host in host_list:
        tmp_result = []
        tmp_result.append(link_delay)
        tmp_result.append(r_host)
        # test for pingTestNum times
        for _ in range(test_rounds):
            ping_cmd = 'ping -c 1 %s' % exp.get_node(r_host).IP()
            ping_result = exp.get_node('h1').cmd(ping_cmd)
            # return None if no position in the string matches
            match = pattern.search(ping_result)
            if match:
                tmp_result.append(match.group(1))
            else:
                logger.info('%s is unreachable from host 1', r_host)
                tmp_result.append('x')  # x for unavailable

        logger.debug('--- end latency test to host: %s', r_host)

        # save ping results in csv
        with open(csv_file_path, 'a') as csv_file:
            csv_writer = csv.writer(csv_file, quoting=csv.QUOTE_NONE)
            csv_writer.writerow(tmp_result)
    logger.info('latency test finished')
    # -------------------------------------------

    logger.info('wait 2 seconds and stop the experiment')
    sleep(2)
    logger.info('stop current experiment')
    exp.stop()


def loss_test(cluster, topo, test_para, data_path):
    """Loss rate test between h1 and other hosts

    Use ping command, each round only ping once

    Args:
        cluster (maxinet.Cluster):
        topo (mininet.Topo): network topology
        test_para (tuple): (test_rounds, ping number of each round)
        data_path (str): dir for saving results as csv
    """

    host_list = topo.hosts()
    host_num = len(host_list)
    link_loss = topo.get_paradict()['link_loss']

    # creat experiment
    exp = maxinet.Experiment(cluster, topo)
    exp.setup()
    logger.info('loss test with link_loss:%d is setup', link_loss)

    # save node distribution
    sv_dist_node(exp, 'losstest', data_path)

    pattern = re.compile(r"([\d]*)\% packet")

    # === run loss test ===
    # -- ping all round
    logger.info('start ping all round')
    for s_host in host_list:
        # loop for recv host
        for r_host in host_list:
            # ping until success for warming up
            ping_flag = 1
            ping_cmd = 'ping -c 1 %s' % exp.get_node(r_host).IP()
            while ping_flag:
                ping_result = exp.get_node(s_host).cmd(ping_cmd)
                match = pattern.search(ping_result)
                if int(match.group(1)) == 0:
                    ping_flag = 0
        logger.debug('--- end ping all of %s', s_host)
    logger.info('finish ping all')

    # run ping and process the result with matching pattern
    # csv data file
    csv_file_name = 'losstest_hn%d_wn%d.csv'\
        % (host_num, exp.cluster.num_workers())
    csv_file_path = path.join(data_path, csv_file_name)

    logger.info('start loss test from h1')
    host_list.remove('h1')  # remove h1 from list

    # loss rate test from host1
    # -------------------------------------------
    for r_host in host_list:
        tmp_list = []
        tmp_list.append(link_loss)
        tmp_list.append(r_host)
        for _ in range(test_para[0]):
            # test for pingTestNum times
            ping_cmd = 'ping -c %d %s' % (test_para[1],
                                          exp.get_node(r_host).IP())

            # ping from host 1
            ping_result = exp.get_node('h1').cmd(ping_cmd)
            # return None if no position in the string matches
            match = pattern.search(ping_result)
            if match:
                tmp_list.append(match.group(1))
            else:
                logger.info('%s is unreachable from host 1', r_host)
                tmp_list.append('x')  # x for unavailable

        # save ping results in csv
        with open(csv_file_path, 'a') as csv_file:
            csv_writer = csv.writer(csv_file, quoting=csv.QUOTE_NONE)
            csv_writer.writerow(tmp_list)
        logger.debug('--- end loss test with host: %s', r_host)

    logger.info('loss rate test finish')
    # -------------------------------------------

    logger.info('wait 2 seconds and stop the experiment')
    sleep(2)
    logger.info('stop current experiment')
    exp.stop()
