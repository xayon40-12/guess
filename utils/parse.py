#!/usr/bin/env python3

import sys
from parsec import *

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
    def __init__(self, t, spec, name, moms):
        self.t = t
        self.spec = spec
        self.name = name
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
        case [t,n,m]:
            return Line(t,None,n,m)
        case [t,s,n,m]:
            return Line(t,s,n,m)

class Data:
    def __init__(self, t, val):
        self.t = t
        self.val = val
    
    def __str__(self):
        return "Data(t: " + ", val: " + str(self.val) + ")"
    
    def __repr__(self):
        return self.__str__()
    
    def __add__(self, other):
        return Data(self.t+other.t, self.val+other.val)
    
num = regex(r"\d+(\.\d+)?(e-?\d+)?").parsecmap(float)
name = regex(r"[^|\n]+")
vec = sepBy1(num, string(";")).parsecmap(single)
value = (num + (string(":") >> vec)) ^ vec
array = sepBy1(value, string(" ")).parsecmap(single)
moms = sepBy1(array, string("/")).parsecmap(single)
p = string("|")
nl = string("\n")
sline = (num + (p >> name) + (p >> moms) << nl).parsecmap(compact(3)).parsecmap(toLine)
lline = (num + (p >> name) + (p >> name) + (p >> moms) << nl).parsecmap(compact(4)).parsecmap(toLine)
line = sline ^ lline
lines = many1(line)

def parse(s):
    ls = (lines << eof()).parse(s)
    fields = {}
    for l in ls:
        n = l.spec_name()
        d = Data([l.t], [l.moms])
        if n in fields:
            fields[n] += d
        else:
            fields[n] = d
    return fields

def parse_files(files):
    obs = {}
    for f in files:
        with open(f, 'r') as o:
            obs[f] = parse(o.read())
    return obs
        

def main() -> int:
    if len(sys.argv) == 1:
        ls = "".join(sys.stdin.readlines())
        print(parse(ls))
    else:
        print(parse_files(sys.argv[1:]))
    return 0

if __name__ == '__main__':
    sys.exit(main())
