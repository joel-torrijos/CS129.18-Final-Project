# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 17:23:35 2016

@author: Jose Emmanuel
"""

from mlp import MultilayerPerceptron

def main():
    topology = [3,3,2]
    brain = MultilayerPerceptron(topology)
    
    brain.printNetwork()
    data = [0.8,0.9,0.8]
    target = [0.9,0.5]
    
    idealError = 0.5
    brain.feedForward(data)
    print "After feed forward"
    brain.backPropagation(target)
    brain.printNetwork()
    
main()
    