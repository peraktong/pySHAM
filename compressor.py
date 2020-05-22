import numpy as np
import h5py
import math
import gzip
import os
import pickle
import sys
import time

time_start = time.time()


def generic_open(f, buffering=100000000):
    """
    Returns
    -------
    fp : file handle
    need_to_close : bool
    """
    if hasattr(f, 'read'):
        return f, False
    else:
        if f.endswith('.gz'):
            fp = gzip.open(f, 'r')
        else:
            fp = open(f, 'r', int(buffering))
        return fp, True

inputs = sys.argv
input_file = inputs[1]
output_file = inputs[2]

#input_file = "hlist_1.00000.list"
#output_file = "hlist.h5"
f, need_to_close = generic_open(input_file)
count = 0
data = {}
for line in f:
    if count%100000==0:
        print("reading line %d"%count)
    line = line.split()
    if count==0:
        keys=line
        for k in keys:
            data[k]=[]
    #print(len(line))
    if len(line) == len(keys) and count > 1:
        for i,k in enumerate(keys):
            data[k].append(float(line[i]))
    """
    if count>1000:
        break
    
    """

    count+=1
print("Finish reading file")

# save:
hf = h5py.File(output_file, 'w')
for i,k in enumerate(keys):
    hf.create_dataset(str(k), data=data[k], dtype="f")
hf.close()
print("Allset")
print("time we use %.2f seconds"%(time.time()-time_start))