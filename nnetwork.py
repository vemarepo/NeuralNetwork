import logging
from edge import Edge
from node import InputNode, InputBiasNode, HiddenNode, HiddenBiasNode, OutputNode


class NNetwork:
    """
    We implement a neura network with back propagation
    INPUTS:
    ninputs: Number of inputs
    noutputs: Number of outputs
    nlayers: Number of hidden layers
    type_network: Type of Network
    """
    def __init__(self, ninputs, noutputs, nlayers, type_network="FULLYCONNECTED" ):
        self.ninputs = ninputs
        self.noutputs = noutputs
        self.nlayers = nlayers
        self.nodesdict  = {}
        self.edgesdict  = {}
        self.nodeslayer  = {} #Stores the network by layers
        self.inputedgesdict  = {} ##Stores the input edges key is the node id, val is list of input nodes to node
        self.outputedgesdict  = {} ##Stores the input edges key is the node id, val is list of input nodes to node
        self.logger = logging.getLogger("Network")
        self.logger.info("Init Network")
        self.layeridlist = []  # Stores a list of layer ids
        self.inpnodeids  = []  # Stores a list of input node ids
        self.outpnodeids  = [] # Stores a list of output node ids

    def create_network(self,):
        pass

    def create_node(self, ln):
        nodetype, nodeid, layer = ln[1].upper(), int(ln[2]),int(ln[3])
        if nodetype == "I":
            node = InputNode(nodeid, layer)
            self.inpnodeids.append( nodeid )
        elif nodetype == "IB":
            node = InputBiasNode(nodeid, layer)
            self.inpnodeids.append( nodeid )
        elif nodetype == "H":
            node = HiddenNode(nodeid, layer)
        elif nodetype == "HB":
            node = HiddenBiasNode(nodeid, layer)
        elif nodetype == "O":
            node = OutputNode(nodeid, layer)
            self.outpnodeids.append( nodeid )

        self.nodesdict[nodeid]  = node

        if self.nodeslayer.has_key(layer):
            self.nodeslayer[layer].append(node)
            self.logger.debug( " Existing Layer  %d  Adding Node  %s"%(layer,  node.nodeid_stg) )

        else:
            self.nodeslayer[layer] = [node,]
            self.logger.debug( " New Layer  %d  Adding Node %s"%(layer,  node.nodeid_stg) )
            self.layeridlist.append(layer)

    def create_edge(self, ln):
        """ Create an edge between two nodes """
        nodeid_i, nodeid_j =  int(ln[1]),int(ln[2])
        edge = Edge(nodeid_i, nodeid_j)
        self.edgesdict[(nodeid_i, nodeid_j)]  = edge
        if self.inputedgesdict.has_key(nodeid_j):
            self.inputedgesdict[nodeid_j].append(edge)
        else:
            self.inputedgesdict[nodeid_j] = [edge,]

        if self.outputedgesdict.has_key(nodeid_i):
            self.outputedgesdict[nodeid_i].append(edge)
        else:
            self.outputedgesdict[nodeid_i] = [edge,]


    def set_data(self, inputs, targets):
        self.set_inputs( inputs )
        self.set_outputs( targets )

    def set_inputs(self, inputs):
        # inputs is a list
        for nodeid in xrange(self.ninputs):
            node = self.nodesdict[nodeid]
            node.set_val( inputs[nodeid])

    def set_outputs(self, targets):
        # targets is a vectors of outputs
        for i, nodeid in enumerate(self.outpnodeids):
            node = self.nodesdict[nodeid]
            node.set_val( targets[i])

    def get_edgesat_node(self, nodeid):
        if self.inputedgesdict.has_key( nodeid):
            inpedges = self.inputedgesdict[nodeid]
        else:
            inpedges = None
            self.logger.debug( "Node %s has no input edges"%(nodeid))

        if self.outputedgesdict.has_key( nodeid):
            outpedges = self.outputedgesdict[nodeid]
        else:
            outpedges = None
            self.logger.debug( "Node %s has no output edges"%(nodeid))

        return inpedges, outpedges

    def forward_pass(self):
        " Implement the forward pass learning method"""
        layervec = sorted(self.layeridlist)
        for layerid in layervec:
            self.logger.debug( "Forward Pass at Layer %d "%(layerid))
            if self.nodeslayer.has_key(layerid):
                self.logger.debug( " Layerid %d has nodes  %d "%(layerid, len(self.nodeslayer[layerid]) ))

            for node in self.nodeslayer[layerid]:
                self.logger.debug( "Forward Pass at %s  "%(node.nodeid_stg))
                inpedges, outpedges = self.get_edgesat_node(node.nodeid)
                node.forward_pass(inpedges, outpedges)



    def backward_pass(self):
        "Backward of the learning method is implemented here"""
        layervec = sorted(self.layeridlist, reverse=True)
        for layerid in layervec:
            self.logger.debug( "Backward Pass at Layer %d "%(layerid))
            if self.nodeslayer.has_key(layerid):
                self.logger.debug( "Layerid %d has nodes  %d "%(layerid, len(self.nodeslayer[layerid]) ))

            for node in self.nodeslayer[layerid]:
                if node.nodetype != "I" or node.nodetype != "IB" or node.nodetype != "HB":
                    self.logger.debug( "Backward Pass at %s  "%(node.nodeid_stg))
                    inpedges, outpedges = self.get_edgesat_node(node.nodeid)
                    node.backward_pass(inpedges, outpedges)

    def compute_error(self):
	""" Computing the Error """
        self.error = 0.0;
        for nodeid in self.outpnodeids:
            self.error += 0.5*self.nodesdict[nodeid].errorsq2
        #self.logger.critical( "Error: %f "%(self.error))

    def getoutputs(self):
        return [self.nodesdict[nodeid].outval for nodeid in self.outpnodeids]

