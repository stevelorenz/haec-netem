#!/usr/bin/python2

#
# This example shows how to use MaxiNet's CommandLineInterface (CLI).
# Using the CLI, commands can be run interactively at emulated hosts.
# Thanks to our build-in py command you can dynamically change the
# topology.
#

from MaxiNet.Frontend import maxinet
from MaxiNet.tools import FatTree

topo = FatTree(4, 10, 0.1)
cluster = maxinet.Cluster()

exp = maxinet.Experiment(cluster, topo)
exp.setup()

print(exp.get_node('h1').IP())

exp.CLI(locals(), globals())

exp.stop()
