import networkx as nx
import networkx.algorithms.community as community
import time
import matplotlib.pyplot as plt

class A:

    def __init__(self, s):
        self.s = s
        self.c = "123"


class B(A):

    def __init__(self, s):
        super().__init__(s=s)
        self.c = "hei"



a = A(1)
print(a.s)
print(a.c)

b = B(1)
print(b.s)
print(b.c)