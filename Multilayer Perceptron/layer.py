# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 22:22:25 2016

@author: Jose Emmanuel
"""

class Layer:
    
    def __init__(self):
        self.mNeurons = []
    
    def addNeuron(self,n):
        self.mNeurons.append(n)
    
    def getNeurons(self):
        return self.mNeurons
    
    def setNeurons(self,neurons):
        self.mNeurons = neurons
        
    def getInputs(self):
        inputs = [n.value for n in self.mNeurons]
        return inputs
    
    def getWeights(self,c):
        weights = [n.outputWeights[c].value for n in self.mNeurons]
        return weights
        