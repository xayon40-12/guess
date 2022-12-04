#!/usr/bin/env python3

import sys
from os.path import isdir,isfile
from parsec import *
import numpy as np

def compact(n):
    def aggregate(x):
        a = []
        for i in range(n-1):
            a = [x[1]]+a
            x = x[0]
        a = [x]+a
        return a
    return aggregate

def single(v):
    if len(v) == 1:
        return v[0]
    else:
        return v

class Line:
    def __init__(self, t, spec, name, count, moms):
        self.t = t
        self.spec = spec
        self.name = name
        self.count = count
        self.moms = moms
    
    def spec_name(self):
        match self.spec:
            case None:
                return self.name
            case s:
                return self.name + "." + self.spec
    
    def __str__(self):
        return "Line(t: " + str(self.t) + ", spec: " + str(self.spec) + ", name: " + str(self.name) + ", moms: " + str(self.moms) + ")"
    
    def __repr__(self):
        return self.__str__()

def toLine(a):
    match a:
        case (((t, (spec, name)), count), moms):
            return Line(t, spec, name, count, moms)

class Data:
    def __init__(self, t, count, val):
        self.t = t
        self.count = count
        self.val = val
    
    def __str__(self):
        return "Data(t: " + ", count" + str(self.count) + ", val: " + str(self.val) + ")"
    
    def __repr__(self):
        return self.__str__()
    
    def __add__(self, other):
        return Data(self.t+other.t, self.count+other.count, self.val+other.val)

none = string("").result(None)
num = regex(r"\d+(\.\d+)?(e-?\d+)?").parsecmap(float)
name = regex(r"[^|\n]+")
vec = sepBy1(num, string(";"))
value = ((num + (string(":") >> vec)) ^ (none + vec)).parsecmap(compact(2))
array = sepBy1(value, string(" "))
moms = sepBy1(array, string("/"))
p = string("|")
nl = string("\n")
nup = num << p
nap = name << p
nuh = num << string("#")
line = (nup + (count(nap, 2) ^ (none + nap)) + (nuh ^ none) + (moms << nl)).parsecmap(toLine)
lines = many1(line)

def moms2cols(moms):
    if moms[0][0][0] == None:
        cols = [[i]+[float(v) for v in vs[1]] for i,vs in enumerate(moms[0])]
        coord = None
    else:
        cols = [[float(vs[0])]+[float(v) for v in vs[1]] for vs in moms[0]]
        coord = [vs[0] for vs in moms[0]]
    def remove_coord(m):
        if coord == None:
            return [vs[1] for vs in m]
        elif [vs[0] for vs in m] == coord:
            return [[float(v) for v in vs[1]] for vs in m]
        else:
            print("Coordinates are different.")
            exit(-1)
            
            
    for m in moms[1:]:
        cols = [c+m for c,m in zip(cols,remove_coord(m))]
    return cols

def parse(s):
    ls = (lines << eof()).parse(s)
    fields = {}
    for l in ls:
        n = l.spec_name()
        d = Data([l.t], [l.count], [moms2cols(l.moms)])
        if n in fields:
            fields[n] += d
        else:
            fields[n] = d
    return fields

def near(a,v):
    return min(enumerate(a), key=lambda x: abs(x[1]-v))

def extract(vals, path):
    res = []
    [name, time, coord] = path + (3-len(path))*["*"]
    def extract_coord(val):
        if coord == '*':
            return val
        else:
            (i,c) = near([v[0] for v in val],float(coord))
            return [val[i]]

    def extract_time(data):
        if time == '*':
            res = []
            for t,v in zip(data.t,data.val):
                res = res+[[float(t)]+v for v in extract_coord(v)]
            return res
        else:
            (i,t) = near(data.t,float(time))
            return [[float(t)]+v for v in extract_coord(data.val[i])]
            
    if name == '*':
        for (n,v) in vals.items():
            times = extract_time(v)
            return [[n]+ts for ts in times]
    else:
        return [[name]+ts for ts in extract_time(vals[name])]

def search_file(path):
    names = path.split("/")
    d = names[0]
    names = names[1:]
    while isdir(d):
        d += "/"+names[0]
        names = names[1:]
    if isfile(d):
        return (d, names)
    else:
        print("No file found in path \""+path+"\".")
        exit(-1)

def parse_file(file):
    (f, path) = search_file(file)
    with open(f, 'r') as o:
        return extract(parse(o.read()), path)


def main() -> int:
    if len(sys.argv) == 1:
        print("Nothing to do.")
        return -1
    else:
        for l in parse_file(sys.argv[1]):
            print(' '.join([str(v) for v in l]))
        return 0

if __name__ == '__main__':
    sys.exit(main())
