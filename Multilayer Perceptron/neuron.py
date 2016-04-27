# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 20:51:57 2016

@author: Jose Emmanuel
"""

from weight import Weight
import random
import math

 #self.layers.addNeuron(Neuron(isBias,isInput,isHidden,num_outputs,neuron_num))

class Neuron:
    def __init__(self,flagBias,isInput,isHidden,numOutputs,ide):
        #self.value = value
        self.ide = ide #id
        self.isInput = isInput
        #self.isOutput = isOutput
        self.isHidden = isHidden
        self.numOutputs = numOutputs
        self.flagBias = flagBias
        self.outputWeights = []
        self.gradient = 0;
        for i in range(0,numOutputs):
            randomNum = self.rand(-1,1)
            x = Weight(randomNum,0)
            self.outputWeights.append(x)
        
        if(self.flagBias):
            self.value = 1
        else:
            self.value = 0
            
            
    def updateInputWeights(self,previousLayerNeurons):
        eta = 0.15
        alpha = 0.5
        for i in range(0,len(previousLayerNeurons)-1):
            n = previousLayerNeurons[i]
            oldDeltaWeight = n.outputWeights[self.ide].deltaValue
            newDeltaWeight = 0.15 * n.value * self.gradient + 0.5 * oldDeltaWeight
            
            n.outputWeights[self.ide].deltaWeight = newDeltaWeight
            n.outputWeights[self.ide].deltaWeight  += newDeltaWeight
            
            #n.outputWeights[self.ide] = w
            #previousLayerNeurons[i] = n
            
        return previousLayerNeurons
    
    def calculateOutputGradient(self,targetVal):
        delta = targetVal - self.value
        self.gradient = delta * self.tanhDerivative(self.value)
    
    def calculateHiddenGradient(self,neurons):
        dow = self.sumDOW(neurons)
        self.gradient = dow * self.tanhDerivative(self.value)
    
    def sumDOW(self,neurons):
        sum = 0
        for i in range(len(neurons)-1):
            sum += self.outputWeights[i].value * neurons[i].gradient
        
        return sum
    
    def tanhDerivative(self,val):
        return 1.0 - val * val
    
    def activate(self,val):
        return math.tanh(val)
    
    def feedForward(self,inputs, weights):
        newVal = 0
        for i in range(0,len(inputs)):
            newVal += inputs[i] * weights[i]
        
        self.value = self.activate(newVal)
        
    def getWeights(self):
        return self.outputWeights
    
    def rand(self,min,max):
        return random.uniform(min,max)
        
    def printWeights(self):
        for i, ow in enumerate(self.outputWeights):
            #print i
            temp = self.outputWeights[i].value
            print "[" + str(i) + "]" + str(temp)
    
    def getOutputValue(self):
        return self.value

    
    
    
        

'''
sList = []
x = Weight(3,4)
sList.append(x)
x = Weight(4,6)
sList.append(x)
x = Weight(5,10)
sList.append(x)

newList = [n.value for n in sList]

print "PRO WAY"
for i in range(0,len(newList)):
    print newList[i]

print "NOOBWAY"
for i in range(0,len(sList)):
    print sList[i].value

sList[0].value = 100

for i in range(0,len(sList)):
    print sList[i].value
'''
    


