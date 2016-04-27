# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 16:28:10 2016

@author: Jose Emmanuel
"""
from layer import Layer
from neuron import Neuron
import math

class MultilayerPerceptron:
    def __init__(self, topology):
        self.setTopology(topology)
        num_layers = len(self.topology)
        self.layers = []
        
        for i in range (0,num_layers):
            x = Layer()
            self.layers.append(x)
            if(i == len(self.topology)-1):
                num_outputs = 0
            else:
                temp = i +1
                print temp
                print topology[temp]
                num_outputs = topology[i+1]
            
            for neuron_num in range(topology[i]+1):
                isBias = False
                isInput = False
                isHidden = False
                
                if(i == 0):
                    isInput = True
                if(neuron_num == topology[i]):
                    isBias = True
                if(i != (len(self.topology)-1) and i != 0):
                    isHidden = True
                
                n = Neuron(isBias,isInput,isHidden,num_outputs,neuron_num)
                self.layers[len(self.layers)-1].addNeuron(n)
                            
    def feedForward(self,inputVals):
        for i in range(0,len(inputVals)):
            self.layers[0].mNeurons[i].outputValue = inputVals[i]
        
        self.layers[0].mNeurons[len(inputVals)].outputValue = 1
        
        for i in range (1,len(self.layers)):
            previousLayer = self.layers[i-1]
            
            for j in range (0,len(self.layers[i].getNeurons())-1):
                inputs = previousLayer.getInputs()
                weights = previousLayer.getWeights(j)
                
                self.layers[i].mNeurons[j].feedForward(inputs,weights)
    
    def printOutputLayer(self):
        for i in range(0,len(self.topology)-1):
            outputNeuron = self.layers[len(self.layers)-1].getNeurons()[i]
            print "OUTPUT NEURON " + i + ": "  + outputNeuron.getOutputVale()

    def backPropagation(self,targetVals):
        outputLayer = self.layers[len(self.layers)-1]
        overallNetError = 0.0
        for n in range(0,len(outputLayer.getNeurons())-1):
            delta = targetVals[n] - outputLayer.getNeurons()[n].getOutputValue()
            overallNetError += delta * delta
        
        overallNetError /= len(outputLayer.getNeurons()) -1
        
        overallNetError = math.sqrt(overallNetError)
        
        #recentAverageError = (recentAverageError * recentAverageSmoothingFactor + overallNetError)/ (recentAverageSmoothingFactor + 1.0)
        
        for n in range(0,len(outputLayer.getNeurons())-1):
            outputLayer.mNeurons[n].calculateOutputGradient(targetVals[n])
        
        for layer_index in range(len(self.layers)-2,0,-1):
            hiddenLayer = self.layers[layer_index]
            nextLayer = self.layers[layer_index+1]
            
            for n in range(0,len(hiddenLayer.getNeurons())-1):
                hiddenLayer.mNeurons[n].calculateHiddenGradient(nextLayer.getNeurons())
                
        for layer_index in range(len(self.layers)-1,0,-1):
            layer = self.layers[layer_index]
            previousLayer = self.layers[layer_index-1]
            
            previousLayerNeurons = previousLayer.mNeurons
            
            for n in range(0,len(layer.mNeurons)-1):
                layer.mNeurons[n].updateInputWeights(previousLayerNeurons)
            
        
    def getResults(self):
        resultVals = []
        
        for i in range(0,len(self.layers[len(self.layers)-1].neurons)):
            resultVals.append(self.layers[len(self.layers)-1].neurons[i].getOutputValue())
        
        return resultVals
    
    def setTopology(self,topology):
        self.topology = topology
    
    def setLayers(self,layers):
        self.layers = layers
    
    def getTopology(self):
        return self.topology
    
    def getLayers(self):
        return self.layers
    
    def printNetwork(self):
        #ow in enumerate(self.outputWeights):
        for i, t in enumerate(self.topology):
            print "==========="
            print "Layer: " + str(i)
            print "==========="
            mNeurons = self.layers[i].mNeurons
            for j in range(self.topology[i]+1):
                print "Neuron: " + str(mNeurons[j].ide) + " Value: " + str(mNeurons[j].value)
                mNeurons[j].printWeights()
            print " "
            
    