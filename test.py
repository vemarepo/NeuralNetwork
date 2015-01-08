from nnetwork import NNetwork
import sys
import pandas
import numpy as np
import logging
import logging.config
module_logger = logging.getLogger("BaseLogger")

def getdata():
    np.random.seed(42)
    x0 = np.random.randn(100)
    x1 = np.random.randn(100)
    bias = np.ones(100)
    inputs  = [(x, y, b ) for x, y, b in zip(x0, x1, bias)]
    y0 = np.sin(x0)*np.sin(x0) + np.cos(x1)*np.cos(x1)
    y1 = np.sin(x1)*np.sin(x1) + np.cos(x0)*np.cos(x0)
    outputs  = [(x, y ) for x, y in zip(y0, y1)]
    return inputs, outputs


def create_fullyconnected_network():
    X = pandas.read_csv( "layout_example_bias.txt", header=None)
    X.columns = ["nelem", "nelemtype", "id1", "id2"]
    subX = X[X.nelem == "N"]
    ninputs  =  np.logical_or(subX.nelemtype == "I" , subX.nelemtype == "IB" ).sum()
    noutputs = (subX.nelemtype == "O").sum()
    nlayers = int( X[X.nelem == "N"]["id2"].max())
    print ninputs, noutputs, nlayers
    N = NNetwork(ninputs, noutputs, nlayers, type_network="FULLYCONNECTED")
    with   open("layout_example_bias.txt") as f:
        lines = f.readlines()
        for line in lines:
            if ( len(line ) != 0) and (line[0] != "#"):
                ln = line.strip().split(",")
                #print ln
                if ln[0] == "N": #Type, id, layernums
                    N.create_node( ln)
                elif ln[0] == "E": #Type, id, layernums
                    N.create_edge( ln)

    return N


logging.config.fileConfig(logfnm)

N = create_fullyconnected_network()
inputs, targets = getdata()

for epoch in xrange(1000):
    agg_error = 0.0
    for inp, targ in zip(inputs, targets):
        #print inp, targ
        N.set_data(inp, targ)
        N.forward_pass()
        N.backward_pass()
        N.compute_error()
        op = [N.nodesdict[nodeid].outval for nodeid in N.outpnodeids]
        
        wts = [edge.wt for ky, edge in N.edgesdict.iteritems()]
        agg_error += N.error
    print "Net Error after epoch", epoch, np.sqrt(agg_error/len(inputs))
    

agg_error = 0.0
for inp, targ in zip(inputs, targets):
    N.set_data(inp, targ)
    N.forward_pass()
    N.compute_error()
    agg_error += N.error

print "Net Error after epoch",  np.sqrt(agg_error/len(inputs))

