#!/usr/local/anaconda/bin/python2.7

# Code from Chapter 4 of Machine Learning: An Algorithmic Perspective
# by Stephen Marsland (http://seat.massey.ac.nz/personal/s.r.marsland/MLBook.html)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008

import sys, os, rbf
from numpy import *

data = loadtxt('winequality-red-headless.csv',delimiter=';')
#data = loadtxt('winequality-white-headless.csv',delimiter=';')

# 1 column with targets
# target = zeros((shape(data)[0],1))

def genKfold(data):
        kf_traint = []
        kf_testt = []
        kf_train = []
        kf_test = []
        for k in range(1,5):
                kfold_train = []
                kfold_test = []
                kfold_traint = []
                kfold_testt = []
                row = 0
                cnt = 1
                for e in data:
                        if cnt == k:
                            kfold_test.append(e[0:11])
                            kfold_testt.append(e[11:12])
                        else:
                            kfold_train.append(e[0:11])
                            kfold_traint.append(e[11:12])
                        cnt += 1
                        if (cnt > 5):
                                cnt = 1
                        row += 1
                kf_train.append(kfold_train)
                kf_test.append(kfold_test)
                kf_traint.append(kfold_traint)
                kf_testt.append(kfold_testt)
                kt = asarray(kf_train)
                ktt = asarray(kf_traint)
                ks = asarray(kf_test)
                kst = asarray(kf_testt)
        return (kt, ktt, ks, kst)
        

def findParameters(trials = 100):
        for runs in range(1,trials + 1):
                net = rbf.rbf(train,traint,10,0,0)
                net.rbftrain(train,traint,0.00001 + runs * 0.00001,2000)
                output = net.rbffwd(alldata)
                err = average(abs(output - alltgts))
                print "Params: (" + str(0.00001 + runs * 0.00005) + ", 2000) / Error: " + str(err)

                net = rbf.rbf(train,traint,10,0,0)
                numruns = 2000 + runs * 10
                net.rbftrain(train,traint,0.00001,numruns)
                output = net.rbffwd(alldata)
                err = average(abs(output - alltgts))
                print "Params: (0.00001, " + str(numruns) + ") / Error: " + str(err)

def findRbfs(trials = 100):
        for runs in range(1,trials + 1):
                nrbfs = 1
                net = rbf.rbf(train,traint,nrbfs + runs,0,0)

                net.rbftrain(train,traint,0.00001,2000)
                output = net.rbffwd(alldata)
                err = average(abs(output - alltgts))
                print "Num of RBFS: " + str(nrbfs + runs) + " / Error: " + str(err)


def predict(train, traint, test, testt):

        for res in range(0,2000):
                tgts = []
                for runs in range(0,20):
                        net = rbf.rbf(train,traint,16,0,0)
                        nbf = 9000
                        net.rbftrain(train,traint,0.00001,nbf)
                        output = net.rbffwd(alldata)
                        err = average(abs(output - alltgts))

                        tgts.append(str(14) + '\n' + str(1599) + str('\n'))
                        #f = open('res-' + str(runs) + '.csv','w')
                        #f.write("Error for this run: " + str(err) + "\n")
                        cnt = 0
                        for e in output:
                                #for i in range(1,20):
                                #       f.write(str(e[0]) + str(','))
                                #f.write('\n')
                                tgts.append(str(average(e)) + str('\n'))
                                cnt += 1

                        #f.close()
                f = open('tgts-' + str(err)[0:5] + '.csv','w')
                for each in tgts:
                        f.write(each)
                f.close()

def predictKfold(train, traint, test, testt, runs):
        err = []
        for res in range(0,runs):
                tgts = []
                row = 0
                for folds in range(0,4):
                        net = rbf.rbf(train[folds],traint[folds],16,0,0)
                        nbf = 2000
                        net.rbftrain(train[folds],traint[folds],0.0001,nbf)
                        output = net.rbffwd(test[folds])
                        err.append(average(abs(output - testt[folds])))
                        for e in output:
                                tgts.append(str(e[0]) + str('\n'))
                        row += 1
                print average(err)
        f = open('tgts-' + str(average(err))[0:5] + '.csv','w')
        f.write(str(14) + '\n' + str(1599) + str('\n'))        
        for each in tgts:
                f.write(each)
        f.close()

#findRbfs()
#findParameters()
(train, traint, test, testt) = genKfold(data)
print len(train)
predictKfold(train, traint, test, testt, 20)
