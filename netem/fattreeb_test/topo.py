#!/usr/bin/python2
# -*- coding: utf-8 -*-
"""
# Author : Xiang, Zuo
# Email  : xianglinks@gmail.com
# About  : Build binary fat tree topology

Classes:

FatTreeBinary: binary fat tree topology
"""
import random
import re
from math import log

from mininet.topo import Topo


# === Functions ===

def rand_byte():
    """Generater a random byte"""
    return hex(random.randint(0, 255))[2:]


def make_mac(host_index):
    """Make MAC address for a host

    Args:
        host_index (int): index number of host
    """
    return rand_byte() + ":" + rand_byte() + ":" +\
        rand_byte() + ":00:00:" + hex(host_index)[2:]


def make_dpid(switch_index):
    """Make DPID for SDN switch

    Args:
        switch_index (int): index number of switch
    """
    mac_addr = make_mac(switch_index)
    dp_id = "".join(re.findall(r'[a-f0-9]+', mac_addr))
    return "0" * (12 - len(dp_id)) + dp_id


def make_ip(host_index, ip_class='B'):
    """Make IP address for a host

    Args:
        host_index (int): index number of host
        ip_class (str): class of ip addr, 'B' or 'C'
    """
    # B class ip, net-addr 172.16.0.0
    if ip_class == 'B':
        if host_index <= 0:
            raise ValueError('host number should greater than 0')
        elif host_index <= 255:
            host_ip = '172.16.0.' + str(host_index)
        elif host_index <= 65534:
            host_ip = '172.16.' + str(int(host_index / 256)) + '.'\
                + str(host_index - 256 * int(host_index / 256))
        else:
            raise ValueError('B class IP support maximal 65534 hosts')

    # C class ip, net-addr 192.168.36.0
    if ip_class == 'C':
        if host_index <= 0:
            raise ValueError('host number should greater than 0')
        elif host_index <= 254:
            host_ip = '192.168.36.' + str(host_index)
        else:
            raise ValueError('C class IP support maximal 254 hosts')

    return host_ip


# === Topology  ===

class FatTreeBinary(Topo):
    """Binary fat tree topology


    Default using B class private IP for virtual hosts
    network addr: 172.16.0.0, support maximal 65534 hosts

    Attributes:
        para_dict (dict): dictionary of topology parameters
            host_num, switch_num, host_bw, link_delay, link_loss, link_jitter
    """

    def __init__(self, host_num=2, host_bw=10, delay=10,
                 loss=0, jitter=0, **opts):
        """Init function for FatTreeBinary class

        Args:
            host_num (int): number of leaf hosts
            bw (int): basic bandwidth of hosts
            delay (int): delay for every link
            loss (int): loss rate for every link
            jitter (int): jitter for every link

        Raises:
            ValueError: if maximal link bandwidth larger than 1000Mbps
        """

        Topo.__init__(self, **opts)

        s_index = 1  # init switch index
        new_bw = host_bw
        leaves = []

        # create binary fat tree
        # -------------------------------------------------
        # generate leaves, each host connected to a switch
        for h_index in range(host_num):
            # add host with specified ip
            leaves.append(
                self.addHost('h' + str(h_index + 1),
                             ip=make_ip(h_index + 1),
                             mac=make_mac(h_index + 1))
            )
        todo = leaves  # singel host as leaves

        # connect leaves
        while len(todo) > 1:
            new_todo = []
            # each time connect two switches
            for i in range(0, len(todo), 2):
                # add switch with specified listenPort
                new_switch = self.addSwitch(
                    's' + str(s_index),
                    **dict(listenPort=(13000 + s_index - 1))
                )

                s_index = s_index + 1
                # add switch as new connected layer
                new_todo.append(new_switch)

                # add link with bandwidth and delay
                self.addLink(todo[i], new_switch, bw=new_bw,
                             delay=str(delay) + "ms", loss=int(loss),
                             jitter=str(jitter) + "ms")

                if len(todo) > (i + 1):
                    self.addLink(todo[i + 1], new_switch, bw=new_bw,
                                 delay=str(delay) + "ms", loss=int(loss),
                                 jitter=str(jitter) + "ms")
            todo = new_todo
            new_bw = new_bw * 2
        # -------------------------------------------------

        # define topo parameter dictionary
        self.para_dict = {
            'host_num': host_num,
            'switch_num': len(self.switches()),
            'host_bw': host_bw,
            'link_delay': delay,
            'link_loss': loss,
            'link_jitter': jitter,
        }

        # check if parameter out of range
        if self.get_maxbw() > 1000:
            raise ValueError('warning: the maximum bandwidth \
                             should not larger than 1000 Mbps')

    def get_paradict(self):
        """Get topology parameters dict

        Returns:
            self.para_dict (dict): dict of topology parameters
        """
        return self.para_dict

    def get_treedepth(self):
        """Get the depth of binary fattree

        Returns:
            tree_depth (int): depth of binary fattree
        """
        host_num = self.para_dict['host_num']

        if (log(host_num, 2) - int(log(host_num, 2))) == 0:
            tree_depth = int(log(host_num, 2))
        else:
            # if is not a full tree
            tree_depth = int(log(host_num, 2)) + 1

        return tree_depth

    def get_maxbw(self):
        """Get the maximal bandwith value in the fattree

        the bandwidth of the root switch

        Returns:
            max_bw (int): the maximal bandwith
        """
        tree_depth = self.get_treedepth()
        max_bw = int(self.para_dict['host_bw'] * pow(2, (tree_depth - 1)))
        return max_bw

    def switch_names(self):
        """Get list of names for switches"""
        return ['s' + str(i + 1)
                for i in range(len(self.switches()))]

    def node_names(self):
        """Get list of names for all nodes"""
        return self.hosts() + self.switch_names()
