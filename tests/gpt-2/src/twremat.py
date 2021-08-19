from subprocess import Popen, PIPE
import random
import os
import sys
import tempfile
from tqdm import tqdm

BINDIR=os.path.join(os.path.dirname(sys.argv[0]), 'bin')
TWREMAT=os.path.join(BINDIR, 'twremat')

# Allow users to pass 'humanized' memlimit values as strings.
def parse_memlimit(memlimit):
    if memlimit[-1] == 'K':
        return int(memlimit[:-1]) * 1000
    elif memlimit[-1] == 'M':
        return int(memlimit[:-1]) * 1000000
    elif memlimit[-1] == 'G':
        return int(memlimit[:-1]) * 1000000000
    else:
        return int(memlimit)

def runtwremat(gr, memlimit, target):
    if type(memlimit) is str:
        memlimit = parse_memlimit(memlimit)

    fname = tempfile.mktemp()
    outname = tempfile.mktemp()
    with open(fname, 'w') as fp:
        print('p remat2', file=fp)
        print(f'memlimit {memlimit}', file=fp)
        for (n, info) in gr.items():
            deps = ' '.join(str(d) for d in info['deps'])
            if info['type'] == 'normal':
                cpu = info['cpu']
                mem = info['mem']
                weight = f'cpu {cpu} mem {mem}'
            elif info['type'] == 'effectful':
                weight = 'effectful'
            elif info['type'] == 'pointer':
                weight = 'pointer'
            if n in target:
                tstr = 'target'
            else:
                tstr = ''
            print(f'node {n} deps {deps} {weight} {tstr}', file=fp)
    print(' '.join([TWREMAT, fname, outname]))
    proc = Popen([TWREMAT, fname, outname])
    assert proc.wait() == 0
    out = []
    with open(outname, 'r') as fp:
        for line in fp:
            line = line.split()
            if line and line[0] == 'c':
                out.append(('compute', int(line[1])))
            elif line and line[0] == 'f':
                out.append(('free', int(line[1])))
            elif line:
                print(line)
                exit()
    return out
