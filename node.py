from __future__ import division
import numpy as np
import logging
from edge import Edge

module_logger = logging.getLogger("Node")

def sigmoid(x, lmbd):
    return 1./(1+ np.exp( - lmbd * x ))


def symmetric_sigmoid(x, lmbd):
    return 2*sigmoid(x, lmbd)  - 1


def sigmoid_derivative(x, lmbd):
    y = sigmoid(x, lmbd)
    return y*(1-y)

def symmetric_sigmoid_derivative(x, lmbd):
    return 2*sigmoid_derivative(x, lmbd)

class Node:
    """ Base Class of a neural network node 
    INPUTS:
    nodeid : Id of the node
    layerid: The layer in which this node is present
    """
    
    def __init__(self, nodeid, layerid, nodetype ):
        self.layerid = layerid
        self.nodeid = nodeid
        self.nodetype = nodetype
        self.nodeid_stg = "Lay%d_id%d_type%s"%(self.layerid, self.nodeid, self.nodetype )
        self.netinp = 0.0
        self.outval = 0.0
        self.logger = logging.getLogger("Node")
        self.lmbd = 1.0

    def eval_activate_func(self, x, lmbd=1.):
        return symmetric_sigmoid(x, lmbd)
        
    def eval_derivative_activate_func(self, x, lmbd):
        return symmetric_sigmoid_derivative(x, lmbd)


class InputNode(Node):
    def __init__(self, nodeid, layerid):
        Node.__init__( self, nodeid, layerid, "I")
        self.logger.debug("Creating Input Node Nodeid%s  Layerid%d" %(self.nodeid, layerid))

    def set_val(self, x):
         self.netinp = x
         self.logger.debug("Setting Value %f at Nodeid %s  " %(x, self.nodeid_stg ))

    def activate_func(self, ):
        if  self.nodetype == "I":
            self.outval  = self.netinp


    def forward_pass(self, inpedges, opedges):
        self.activate_func()
        for edge in opedges:
            edge.set(self.outval)
            edge.forward_pass()


    def backward_pass(self, inpedges, opedges):
        pass

class InputBiasNode(Node):
    def __init__(self, nodeid, layerid):
        Node.__init__( self, nodeid, layerid, "IB")
        self.logger.debug("Creating Input Bias Node Nodeid%s  Layerid%d" %(self.nodeid, layerid))

    def set_val(self, x):
        if (x != 1.0):
            self.logger.debug("Setting Value %f at Nodeid %s  " %(1., self.nodeid_stg ))
        self.netinp = 1.0
        self.logger.debug("Setting Value %f at Nodeid %s  " %(1., self.nodeid_stg ))

    def activate_func(self, ):
        if  self.nodetype == "IB":
            self.outval  = self.netinp

    def forward_pass(self, inpedges, opedges):
        self.activate_func()
        for edge in opedges:
            edge.set(self.outval)
            edge.forward_pass()


    def backward_pass(self, inpedges, opedges):
        pass

class HiddenNode(Node):
    def __init__(self, nodeid, layerid):
        Node.__init__(self, nodeid, layerid, "H")
        self.logger.debug("Creating Hidden Node Nodeid%s  Layerid%d" %(nodeid, layerid))
        self.outval = 0
        self.dervval = 0
        self.lmbd = 1.0

    def activate_func(self, ):
        x = self.netinp
        self.outval = self.eval_activate_func(x, self.lmbd)
        self.dervval = self.eval_derivative_activate_func(x, self.lmbd)
        self.logger.debug("Id %s Inp %f OutVal %f DervVal %f" %(self.nodeid_stg, x, self.outval, self.dervval))


    def forward_pass(self, inpedges, opedges):
        self.netinp = sum([edge.o_val for edge in inpedges])
        for edge in inpedges:
            self.logger.debug("ForwPass  Nodeid %s: Inp Val %f  " %(self.nodeid_stg, edge.o_val  ))

        self.activate_func()
        for edge in opedges:
            edge.set(self.outval)
            edge.forward_pass()

    def backward_pass(self, inpedges, opedges):
        self.error =sum([edge.o_bpval for edge in opedges])
        self.bperror = self.error * self.dervval
        for edge in inpedges:
            edge.setbperror( self.bperror)
            edge.backward_pass()
            edge.update_weight()


class HiddenBiasNode(Node):
    def __init__(self, nodeid, layerid):
        Node.__init__(self, nodeid, layerid, "HB")
        self.logger.debug("Creating Hidden Bias Node Nodeid%s  Layerid%d" %(nodeid, layerid))
        self.outval = 0
        self.dervval = 0
        self.lmbd = 1.0

    def activate_func(self, ):
        self.netinp = 1.
        self.outval = 1.0
        self.logger.debug("Id %s Inp %f OutVal %f " %(self.nodeid_stg, 1.0, self.outval))


    def forward_pass(self, inpedges, opedges):
        self.netinp = 1.0
        self.activate_func()
        for edge in opedges:
            edge.set(self.outval)
            edge.forward_pass()

    def backward_pass(self, inpedges, opedges):
         pass

class OutputNode(Node):
    def __init__(self, nodeid, layerid):
        Node.__init__( self, nodeid, layerid, "O")
        self.logger.debug("Creating Output Node Nodeid%s  Layerid%d" %(nodeid, layerid))
        self.target = 0
        self.error = 0
        self.outval = 0
        self.dervval = 0
        self.bperror = 0
        self.lmbd = 2.0

    def activate_func(self, ):
        
        x = self.netinp
        self.outval = self.eval_activate_func(x, self.lmbd)
        self.dervval = self.eval_derivative_activate_func(x, self.lmbd)
        self.error = self.target - self.outval
        self.errorsq2 = self.error * self.error
        self.logger.debug("Id %s Inp %f OutVal %f" %(self.nodeid_stg, x, self.outval))

    def set_val(self, x):
         self.target = x
         self.logger.debug("Setting Target %f at Nodeid %s  " %(x, self.nodeid_stg ))

    def forward_pass(self, inpedges, opedges):
        self.netinp = sum([edge.o_val for edge in inpedges])
        for edge in inpedges:
            self.logger.debug("ForwPass  Nodeid %s: Inp Val %f  " %(self.nodeid_stg, edge.o_val  ))

        self.activate_func()
        
    def backward_pass(self, inpedges, opedges):
        self.bperror = self.error * self.dervval
        self.logger.debug("Backward Pass at Output Nodeid %s Error %f DerVal %f BpErro %f  " %( self.nodeid_stg , self.error, self.dervval, self.bperror))
        for edge in inpedges:
            edge.setbperror( self.bperror)
            edge.backward_pass()
            edge.update_weight()
