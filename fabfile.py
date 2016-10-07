#!/usr/bin/python2
# -*- coding: utf-8 -*-
"""
Fabric is used here to manage the cluster of Workers

# Settings
- Workers are defined as haec_workers roles and setted as default
- Data operations are running parallel, while others use sequential mode

# Add new task
- Write a function with @task decorator in this file to define a new task
- Use @parallel decorator to run task parallel

# Author : Xiang, Zuo
# Email  : xianglinks@gmail.com
"""

from __future__ import with_statement

from re import search
from time import sleep, strftime

from fabric.api import (cd, env, get, local, parallel, put, run, settings,
                        sudo, task)
from fabric.context_managers import hide
from fabric.contrib import project

# Fabric Configurations
# -----------------------------------------------
# number of times for connection
env.connection_attempts = 1
# skip the unavailable hosts
env.skip_bad_hosts = True
env.colorize_errrors = True
# -----------------------------------------------

# Init Remote Hosts and Roles
# -----------------------------------------------
# -- virtual machine test hosts --
VM_LIST = ['mininet@192.156.56.%d:22' % h for h in range(101, 103)]

# -- haec playground workers --
WORKER_LIST = ['odroid@192.168.0.%d:22' % h for h in range(51, 67)]

env.roledefs = {
    'vm_workers': VM_LIST,
    'haec_workers': WORKER_LIST,
    'frondend': 'odroid@192.168.0.30:22'
}

# set default roles, haec_workers
# without -R and -H option, default roles will be used
if not len(env.roles) and not len(env.hosts):
    env.roles = ['haec_workers']

# set password directory
PASSWORD_DICT = {}
for host in VM_LIST:
    PASSWORD_DICT[host] = "mininet"

for host in WORKER_LIST:
    PASSWORD_DICT[host] = "odroid"

env.passwords = PASSWORD_DICT
# -----------------------------------------------


# Operations for MaxiNet
# -----------------------------------------------
@task
@parallel
def del_mxn_cfg():
    """Delete MaxiNet config file"""
    run('rm -f ~/.MaxiNet.cfg')


@task
@parallel
def put_mxn_cfg():
    """Put MaxiNet config file on remote hosts"""
    remote_path = '~/.MaxiNet.cfg'

    # use file for different roles
    if 'haec_workers' in env.roles:
        local_path = './haec_maxinet.cfg'
    elif 'vm_workers' in env.roles:
        local_path = './vm_maxinet.cfg'

    put(local_path, remote_path)


@task
def run_worker():
    """Run MaxiNet WorkerServer process at background without hang up

    Run serially and sleep 1s for each worker.
    Try to make the registration in order and successfully.
    """
    with settings(hide('warnings'), warn_only=True):
        # check if MaxiNetWorker is running
        count = int(run('pgrep -c MaxiNetWorker').splitlines()[0])
        while count == 0:
            # run cmd without pty, stdin data won't be echoed
            sudo('nohup MaxiNetWorker > /dev/null 2>&1 &', pty=False)
            sleep(1)  # wait 1 second
            count = int(run('pgrep -c MaxiNetWorker').splitlines()[0])


@task
@parallel
def kill_worker():
    """Kill MaxiNet WorkerServer process at background"""
    with settings(hide('warnings'), warn_only=True):
        count = int(run('pgrep -c MaxiNetWorker').splitlines()[0])
        while count > 0:
            # kill all MaxiNetWorker
            sudo('killall MaxiNetWorker', pty=False)
            sleep(1)
            count = int(run('pgrep -c MaxiNetWorker').splitlines()[0])
# -----------------------------------------------


# Data & file Operations
# -----------------------------------------------
@task
@parallel
def del_netem_src():
    """Delete haec-netem dir

    This function can be used to remove all test results on remote
    hosts, which need root authority
    """
    sudo('rm -rf ~/haec-netem')


@task
@parallel
def get_results():
    """Get test result directories on Workers

    Local path: /tmp/tmp_result_timestamp
    """
    # check and create local tmp path
    local_path = '/tmp/tmp_result_%s' % strftime('%Y%m%d_%H%M')
    local('mkdir -p %s' % local_path)

    # find all directories with start 'tmp_result_'
    result_dir_lt = str(
        run('find ~/haec-netem/netem -type d -name \'test_result_*\'')
    ).splitlines()

    for result_dir in result_dir_lt:
        get(result_dir, local_path)


@task
@parallel
def sync_src_vm():
    """Sync haec-netem source code on virtual machines, just for testing

    .git and .gitignore will be ignored
    """
    local_dir = '~/src/python-src/haec-netem'
    remote_dir = '~/'  # default in home path
    project.rsync_project(
        remote_dir=remote_dir,
        local_dir=local_dir,
        default_opts='-avczp',
        # add exclude path using script path as root
        exclude=['.git', '.gitignore', 'data_csv'],
        # delete addtional file on remote
        delete=True
    )
# -----------------------------------------------

# (Re)Installation Packages
# -----------------------------------------------
@task
def reinstall_mxn(branch='haec_playground'):
    """Reinstall MaxiNet python lib from forked repository

    Default origin stevelorenz/MaxiNet on github

    Args:
        branch (str): mark the branch to be downloaded
    """

    sudo('rm -rf ~/MaxiNet')
    run('git clone https://github.com/stevelorenz/MaxiNet.git ~/MaxiNet')

    with settings(hide('warnings'), warn_only=True):
        run('git pull origin %s' % branch)
        run('git checkout %s' % branch)
        sudo('make reinstall')


@task
def reinstall_mn(version='2.2.1', option='nfv'):
    """Reinstall Mininet using official install.sh

    Args:
        version (str): version to be installed, default 2.2.1
        option (str): option used for resinstalling mininet
            default: -nfv: mininet, openvswitch and openflow
    """
    sudo('rm -rf ~/openflow && rm -rf ~/mininet')
    run('git clone https://github.com/mininet/mininet.git ~/mininet')

    with cd('~/mininet'):
        run('git checkout -b %s' % version)

    with cd('~/mininet/util'):
        run('bash ./install.sh -%s' % option)


@task
def reinstall_pyro(version='4.43'):
    """Reinstall Pyro4 python module

        Pyro4 is a dependency of MaxiNet
        Current MaxiNet has issues when using latest version of Pyro4
        So the version 4.43 works according to the tests on HAEC Playground
    """
    if not is_package_installed('python-pip'):
        sudo('apt-get install python-pip')
    sudo('pip install Pyro4==%s' % version)


@task
def is_package_installed(pkgname):
    """Check if a pacakge is installed using dpkg

    Args:
        pkgname (str): name of pacakge to be checked
    """
    with settings(hide('output'), warn_only=True):
        output = run('dpkg -s {}'.format(pkgname), pty=False)
        match = search(r'Status: (\w+.)*', output)
        if match and 'installed' in match.group(0).lower():
            return True
        return False
# -----------------------------------------------
