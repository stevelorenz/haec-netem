

1. Turn on Odroids and test connection

    Shell script in ./tools/worker-ctl manage the power of Workers.

    Run ` $ worker-ctl -h ` to get help information, ` $ worker-ctl -p ` will test the connection of each Work.


2. Run SDN Controller

    Default use [POX](https://github.com/noxrepo/pox) Controller, installed at home dir (~/pox).

    Use command ` $ ~/pox/pox.py forwarding.l2_learning ` to run POX using layer 2 learning component.

3. Configuration File

    Config file: ./haec_maxinet.cfg, which will saved as ~/.MaxiNet.cfg on Frontend and Workers.

    Check and edit it according to the [MaxiNet Wiki](https://github.com/MaxiNet/MaxiNet/wiki/Configuration-File). See **remark** 1.


4. Set initial number for GRE tunnels

    MaxiNet of haec_playground branch reads tunnel key from /etc/tunnum file, see **remark** 2.

    According to tests, use 30 as start number (each time after switching on the switch) works fine. ` $ echo 30,30 > /etc/tunnum `

5. Run MaxiNetFrondendServer at the Frontend

    Use command ` $ MaxiNetFrondendServer ` to start the FrontendServer

6. Running WorkerServer process on Workers

    Workers are managed by a python library called [Fabric](http://www.fabfile.org/). To run tasks defined in fabfile.py,
    you need to change directory to the path of this file or use ` -f ` option to explicit file path to load as fabfile

    ` $ fab -l  ` to print possiable commands. `fab -R roles_name` can be used to select the roles for executing commands
    (e.g. ` fab -R haec_workers `). If no -R or -H option is gived, the default roles haec_workers will be used.

    Use ` fab run_worker ` to start WorkerServer process at background on Workers, the registration information should be showed
    by MaxiNetFrondendServer. ` fab kill_worker ` command will kill WorkerServer process.

7. Run network emulation experiments

    Now you can use ` $ MaxiNetStatus ` to check if all Workers are correctly registered. This Cluster can be used for running
    multiple emulation experiments.

    You find all documentation and an API reference here: [MaxiNet API Documentations](https://rawgit.com/MaxiNet/MaxiNet/v1.0/doc/maxinet.html)

8. Shutdown the system and turn off the Workers

    It is highly recommended to shutdown the operating system firstly before switching off the power. Sometime suddenly poweroff can cause
    some problem like Odroid can't successfully boot.

    Use ` $ worker-ctl -s 1-16 ` to shutdown the operating system on all works.
