from math import exp
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from functools import reduce


# takes the neural network and flattens it
def flatten(net):
    flat1 = sum(net, [])
    return sum(flat1, [])


# calculates the output of a node given the total from the dot product of the inputs and the weights
def sigma(num):
    return 1.0/(1.0+exp(-1.0*num))


# calculates the derivative of a node given the total raw input
def sigprime(num):
    sig = sigma(num)
    return sig*(1.0-sig)


# randomly initializes a network with weights
# layerlens is a list of how many nodes are in each hidden layer
# in_len is the number of inputs
# out_len is the number of output nodes
def initnet(layerlens, in_len, out_len):
    network = np.empty(len(layerlens)+1)
    for i in range(len(layerlens)):
        layer = np.empty(layerlens[i])
        if(i==0):
            #special case for first hidden layer
            for k in range(layerlens[i]):
                weights = np.empty(in_len+1)
                for j in range(in_len+1):
                    weights[j]= np.random.uniform(0.0,1.0)
                layer[k] = weights
        else:
            for k in range(layerlens[i]):
                #each node has a number of weights equal to the previous layers number of nodes plus the bias
                weights = np.empty(layerlens[i-1]+1)
                for j in range(layerlens[i-1]+1):
                    weights[j] = np.random.uniform(0.0, 1.0)
                layer[k] = weights
        network[i] = layer
    layer = np.empty(out_len)
    #special case for output layer
    for k in range(out_len):
        #fails with no hidden layer
        weights= np.empty(layerlens[-1]+1)
        for j in range(layerlens[-1]+1):
            weights[j] = np.random.uniform(0.0, 1.0)
        layer[k]=weights
    network[-1] = layer
    #print network
    return network    #return [[[0.268907371619032, 0.945546627056832, 0.263108481983011], [0.219475552278563, 0.226349368036303, 0.69172099025921]], [[0.760209364555744, 0.31324794300938, 0.395712319541701]]]


# takes a network and an example input
# returns the raw output of each node, the sigma function has not been applied
def feed_forward(network, example_in) :
    layers = len(network)
    #instantiate this way so you can use assignment (appending lists to lists gives weird resutls)
    raw_outs = np.empty(layers)
    for i in range(layers):
        if i ==0:
            nodeouts = np.empty(len(network[i]))
            for j in range(len(network[i])):
                # +[1] serves to add the always 1 input of the bias term
                nodeouts[j] = np.dot(network[i][j], example_in+[1])
            raw_outs [i] = nodeouts
        else:
            nodeouts = np.empty(len(network[i]))
            for j in range(len(network[i])):
                # must apply the sigma function in order to feed forward
                nodeouts[j] = np.dot(network[i][j], map(sigma, raw_outs[i-1])+[1])
            raw_outs[i] = nodeouts
    return raw_outs


# takes in a network, the raw outputs of each node (sigma func not yet applied), and an example output
# returns the gradient values (dC/dx) for each node in the network
def nabla(network, raw_outs, example_outs):
    # create a space for the gradients, one for each node in the network
    gradients = [[0 for x in range(len(raw_outs[y]))] for y in range(len(raw_outs))]
    for i in range(len(network))[::-1]:
        for j in range(len(network[i])):
            if i == (len(network) - 1):
                # for the first calculated gradient, the output layer,
                # must use this special function (yout - ytrue) *sigprime(out)
                gradients[i][j] = (sigma(raw_outs[i][j])- example_outs[j]) * sigprime(raw_outs[i][j])
            else:
                # this list comprehension is a list of all weights coming out of the node
                # where the node in question is the one whose gradient is being calculated
                dnext = [network[i+1][x][j] for x in range(len(network[i+1]))]
                # the gradient is the dot product of the weights and
                # the gradients of the next layer times the output of the node
                gradients[i][j] = sigprime(raw_outs[i][j]) * np.dot(dnext, gradients[i+1])
    return gradients


# network is a network of weights to be modified
# gradient is the dc/dx for each node x in each training example
# example_ins is the array of all example inputs in the batch
# raw_outs is the tensor of all raw totals for each node in each example
# the learning rate determines how fast each weight is changed
def modify(network, gradient, example_ins, raw_outs, learning_rate):
    for i in range(len(network)):
        for j in range(len(network[i])):
            for k in range(len(network[i][j])):
                weightchange = 0.0
                for l in range(len(example_ins)):
                    weight_in=0
                    # bias input is always one
                    if k == (len(network[i][j])-1):
                        weight_in =1.0
                    # first layer has inputs from example ins
                    elif i ==0:
                        weight_in = example_ins[l][k]
                    else:
                        #other layers have inputs from the output of the previous layer
                        weight_in = sigma(raw_outs[l][i-1][k])
                    #this is the weight change calculated for this example (not multiplied by learning rate)
                    weightchange += gradient[l][i][j]*weight_in
                # average over number of examples
                avg_weightchange = weightchange/len(example_ins)
                # multiply by learning rate and make negative before change
                network[i][j][k] += -1.0*learning_rate*avg_weightchange
    return network

if __name__ == "__main__":

    print("Please input the length of the input, followed by the number of outputs,"
          " followed by the number of training examples")
    (in_len, out_len, ex_num) =map(int,raw_input().split())
    print("Please enter the name of the file that holds the examples")
    filename = raw_input()
    ex_data = np.empty(ex_num)
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
    #slice the inputs and outputs into separate arrays
    example_ins = [x[0:in_len] for x in ex_data]
    example_outs = [x[in_len:] for x in ex_data]
    #track indices and costs for plotting purposes
    indices =[]
    costs = []
    weights = np.empty(epochs)
    for i in range(epochs):
        gradients = np.empty(ex_num)
        raws_outs = np.empty(ex_num)
        cost_accumulator =0.0
        for j in range(ex_num):
            # calculate the raw outs of every node
            raw_outs = feed_forward(network, example_ins[j])
            raws_outs[j] = raw_outs
            #print raw_outs
            #print network
            # calculate the gradient of each node on this training example
            gradients[j] = nabla(network, raw_outs, example_outs[j])
            # calculate true output of each output node
            outlayer = map(sigma, raw_outs[-1])
            # print outlayer, example_outs[j]
            # calculate cost over each output node and sum
            cost_per_ex = reduce(lambda x, y:x+y, map(lambda x: .5* float(abs(x))**2.0,np.subtract(outlayer,example_outs[j])))
            ##print cost_per_ex
            cost_accumulator += cost_per_ex
        # modify the network weights using the data from this batch
        modify(network, gradients, example_ins, raws_outs, learn_rate)
        # for graphing purposes
        indices.append(i)
        costs.append(cost_accumulator)
        #if i % 10 == 0 :
           # print cost_accumulator
        #for graphing purposes
        weights[i] = flatten(network)
    # for plotting purposes
    indices = indices[1:]
    costs = costs[1:]
    fig1 = plt.figure(1)
    plt.subplot(211)
    plt.plot(indices, costs)
    plt.subplot(212)
    for i in range(len(weights[0])):
        plt.plot(indices, [net[i] for net in weights][1:], label="w"+str(i+1))
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2,ncol=9, mode="expand", borderaxespad=0.)
    print network
    print flatten(network)
    plt.show()

    print("Press enter to quit")
    i = raw_input()
