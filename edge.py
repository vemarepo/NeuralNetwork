import sys
import logging

class Edge:
    def __init__(self, nodeid_i, nodeid_j):
        self.nodeid_i = nodeid_i
        self.nodeid_j = nodeid_j
        self.wt =.1
        #forward pass
        self.i_val = 0.0
        self.o_val = 0.0
        self.delta = 0.0

        #Backward pass back propagation errors
        self.i_bpval = 0.0
        self.o_bpval = 0.0

        #Learning Parameter
        self.gamma = 1.


        self.logger = logging.getLogger("Edge")
        if self.nodeid_i == self.nodeid_j:
            print "Edges on same node id", self.nodeid_i , self.nodeid_j
            sys.exit()

    def forward_pass(self):
        self.o_val = self.wt *self.i_val

    def set(self, val):
        self.i_val  = val

    def setbpdelta(self, val):
        self.delta = val

    def setbperror(self, val):
        self.i_bpval  = val

    def forward_pass(self):
        self.o_val = self.wt *self.i_val

    def backward_pass(self):
        self.o_bpval = self.wt *self.i_bpval

    def update_weight(self):
        self.wtdelta = -self.gamma * self.i_val * self.i_bpval
        self.wt = self.wt - self.wtdelta
