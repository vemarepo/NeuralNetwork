import numpy as np
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

def getdata():
    np.random.seed(42)
    x0 = np.random.randn(100)
    x1 = np.random.randn(100)
    inputs  = [(x, y ) for x, y in zip(x0, x1)]
    y0 = np.sin(x0)*np.sin(x0) + np.cos(x1)*np.cos(x1)
    y1 = np.sin(x1)*np.sin(x1) + np.cos(x0)*np.cos(x0)
    outputs  = [(x, y ) for x, y in zip(y0, y1)]
    return inputs, outputs


inputs, targets = getdata()

net = buildNetwork(2, 3, 2)
ds = SupervisedDataSet(2, 2)
[ds.addSample( xinp, xout) for xinp, xout in zip(inputs, targets)]
trainer = BackpropTrainer(net, ds, verbose=True,)
trainer.module.params[:] = .1* np.ones(len(net.params))
trainer.trainEpochs( 100 )
totalError = 0
for inp, target in ds:
    res = trainer.module.activate(inp)
    e = 0.5 * sum((target-res).flatten()**2)
    totalError += e
    print target, res

print "Total Erroer" , np.sqrt(totalError/len(inputs))
print net.params
