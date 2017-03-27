from math import exp
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from functools import reduce


def sigma(num):
    return 1.0/(1.0+exp(-1.0*num))


def sigprime(num):
    sig = sigma(num)
    return sig*(1.0-sig)


def initnet(layerlens, in_len, out_len):
    network = [None] * (len(layerlens)+1)
    for i in range(len(layerlens)):
        layer = [None] * layerlens[i]
        if(i==0):
            for k in range(layerlens[i]):
                weights = [None] *(in_len+1)
                for j in range(in_len+1):
                    weights[j]= np.random.uniform(0.0,1.0)
                layer[k] = weights
        else:
            for k in range(layerlens[i]):
                weights = [None] *(layerlens[i-1]+1)
                for j in range(layerlens[i-1]+1):
                    weights[j] = np.random.uniform(0.0, 1.0)
                layer[k] = weights
        network[i] = layer
    layer = [None]*out_len
    for k in range(out_len):
        weights= [None] * (layerlens[-1]+1)
        for j in range(layerlens[-1]+1):
            weights[j] = np.random.uniform(0.0, 1.0)
        layer[k]=weights
    network[-1] = layer
    print network
    return network    #return [[[0.268907371619032, 0.945546627056832, 0.263108481983011], [0.219475552278563, 0.226349368036303, 0.69172099025921]], [[0.760209364555744, 0.31324794300938, 0.395712319541701]]]



def feed_forward(network, example_in) :
    layers = len(network)
    raw_outs = [None]*(layers)
    for i in range(layers):
        if i ==0:
            nodeouts =[None]*len(network[i])
            for j in range(len(network[i])):
                nodeouts[j] = np.dot(network[i][j], example_in+[1])
            raw_outs [i] = nodeouts
        else:
            nodeouts = [None] * len(network[i])
            for j in range(len(network[i])):
                nodeouts[j] = np.dot(network[i][j], map(sigma, raw_outs[i-1])+[1])
            raw_outs[i] = nodeouts
    return raw_outs


def nabla(network, raw_outs, example_outs):
    gradients = [[0 for x in range(len(raw_outs[y]))] for y in range(len(raw_outs))]
    for i in range(len(network))[::-1]:
        for j in range(len(network[i])):
            if i == (len(network) - 1):
                gradients[i][j] = (sigma(raw_outs[i][j])- example_outs[j]) * sigprime(raw_outs[i][j])
            else:
                #dnext = map(lambda x: x*sigprime(raw_outs[i][j]),(network[i+1][x][j] for x in range(len(network[i+1]))))
                dnext = [network[i+1][x][j] for x in range(len(network[i+1]))]
                gradients[i][j] = sigprime(raw_outs[i][j]) * np.dot(dnext, gradients[i+1])
    return gradients


def modify(network, gradient, example_ins, raw_outs, learning_rate):

    for i in range(len(network)):
        for j in range(len(network[i])):
            for k in range(len(network[i][j])):
                weightchange = 0.0
                for l in range(len(example_ins)):
                    weight_in=0
                    if k == (len(network[i][j])-1):
                        weight_in =1.0
                    elif i ==0:
                        weight_in = example_ins[l][k]
                    else:
                        weight_in = sigma(raw_outs[l][i-1][k])
                    weightchange += gradient[l][i][j]*weight_in
                avg_weightchange = weightchange/len(example_ins)
                network[i][j][k] += -1.0*learning_rate*avg_weightchange
    return network

print("Please input the length of the input, followed by the number of outputs,"
      " followed by the number of training examples")
(in_len, out_len, ex_num) =map(int,raw_input().split())
print("Please enter the name of the file that holds the examples")
filename = raw_input()
ex_data = [None]*ex_num
infile = open(filename, 'r')
for i in range(ex_num):
    ex_data[i] = map(float, infile.readline().split())
print("Please enter the number of nodes you would like in each hidden layer")
##i.e. if you want 3 layers with 2 nodes each input 2 2 2
layerlens = map(int, raw_input().split())
network = initnet(layerlens, in_len, out_len)
print("How many epochs would you like to perform?")
epochs = int(raw_input())
print("What would you like the learning rate to be?")
learn_rate = float(raw_input())
example_ins = [x[0:in_len] for x in ex_data]
example_outs = [x[in_len:] for x in ex_data]
indices =[]
costs = []
for i in range(epochs):
    gradients = [None]*ex_num
    raws_outs = [None]*ex_num
    cost_accumulator =0.0
    for j in range(ex_num):
        raw_outs = feed_forward(network, example_ins[j])
        raws_outs[j] = raw_outs
        #print raw_outs
        #print network
        gradients[j] = nabla(network, raw_outs, example_outs[j])
        outlayer = map(sigma, raw_outs[-1])
        ##print outlayer, example_outs[j]
        cost_per_ex = reduce(lambda x, y:x+y, map(lambda x: .5* float(abs(x))**2.0,np.subtract(map(float,outlayer),map(float,example_outs[j]))))
        ##print cost_per_ex
        #THE CULPRIT
        cost_accumulator += reduce(lambda x, y:x+y, map(lambda x: .5* float(abs(x))**2.0,np.subtract(map(float,outlayer),map(float,example_outs[j]))))
    modify(network, gradients, example_ins, raws_outs, learn_rate)
    indices.append(i)
    costs.append(cost_accumulator)
    if i % 10 == 0 :
        print cost_accumulator
indices = indices[1:]
costs = costs[1:]
plt.plot(indices, costs)
plt.show()

print("Press enter to quit")
i = raw_input()
