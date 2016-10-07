# HAEC playground NETwork EMulation

Distributed SDN network emulator-[MaxiNet](http://maxinet.github.io/) on HAEC Playground.

Student Research Work at [TU Dresden](https://tu-dresden.de/)

The central Odroid is used as the Frontend and other 16 Odroids on board are Workers.

## Files

## Docs

Read [docs](docs/simple_setup.md)

## Fab Command Usage

1. support arbitrary remote shell commands (without root privilege)

    ` $ fab -R or -H -- [shell command] `, e.g. ` $ fab -R haec_workers -- uname -a` to get kernel info.

2. useful options

        -d    Prints the entire docstring for the given task
        -n    Set number of times to attempt connections
        -P    Sets env.parallel to True, causing tasks and command to run in parallel.
        -t    Set connection timeout in seconds

## Reference Links

- [MaxiNet Homepage](http://maxinet.github.io/#quickstart)
- [MaxiNet Wiki](https://github.com/MaxiNet/MaxiNet/wiki)
- [MaxiNet API Documentations](https://rawgit.com/MaxiNet/MaxiNet/v1.0/doc/maxinet.html)
- [Fabric Documentations](http://docs.fabfile.org/en/1.12/index.html)


## Remark
There are some hardware-related problems for running official MaxiNet on HAEC Playground,
so we have edited some source code. The edited version can be found as **haec_playground** branch in
[stevelorenz/MaxiNet](https://github.com/stevelorenz/MaxiNet) repository.

#### Additional changes:

- Maxinet.cfg

  Add threadpool option: Amount of worker threads to be spawned, see documentation [Pyro4](https://pythonhosted.org/Pyro4/config.html)

  Set threadpool=64: Current version of Odroid-XU4 can not support default 256 pool size,
  64 is here suitable (support totally 31 Workers). Maximal value=124, namely support  61 Workers plus one Frontend.

- Frontend/maxinet.py

  Add codes to read the GRE tunnel key from file("/tmp/tunnum"), because the switch (TP Link T3700G-28TQ) on the HAEC
  may try to stop creation of GRE tunnel with some specified and duplicated key number. And keys should start from usable number and
  also be used only once. So the program will read a key from the file and save the increased number in the same file.


## Contact
Xiang, Zuo

xianglinks@gmail.com
