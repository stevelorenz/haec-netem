#!/bin/bash
# About: Run tests, relax and turn off everything

python2 ./run_test.py  && \
    sleep 5m &&\
    # turn off all workers
    sudo worker-cli -d 1-16 && \
    sudo poweroff
