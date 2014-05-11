#!/usr/local/anaconda/bin/python2.7

# Code from Chapter 4 of Machine Learning: An Algorithmic Perspective
# by Stephen Marsland (http://seat.massey.ac.nz/personal/s.r.marsland/MLBook.html)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008

import sys, os, rbf
from numpy import *

# 1 column with targets
# target = zeros((shape(data)[0],1))

def genKfold(data):
        train = []
        test = []
        train_t = []
        test_t = []
        for k in range(1,6):
                kfold_train = []
                kfold_test = []
                kfold_traint = []
                kfold_testt = []
                row = 1
                cnt = 0
                for e in data:
                        if row == k:
                                kfold_test.append(e[0:11])
                                kfold_testt.append(e[11:12])
                        else:
                                kfold_train.append(e[0:11])
                                kfold_traint.append(e[11:12])
                        row += 1
                        if (row == 6):
                                row = 1
                train.append(asarray(kfold_train))
                train_t.append(asarray(kfold_traint))
                test.append(asarray(kfold_test))
                test_t.append(asarray(kfold_testt))
                
                        
        kt = asarray(train)
        ktt = asarray(train_t)
        ks = asarray(test)
        kst = asarray(test_t)
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

def genConfMtx():
        mtx = [[ 0, 4, 5, 6, 7, 8 ] ,
                [ 4, 0, 0, 0, 0, 0 ] ,
                [ 5, 0, 0, 0, 0, 0 ] ,
                [ 6, 0, 0, 0, 0, 0 ] ,
                [ 7, 0, 0, 0, 0, 0 ] ,
                [ 8, 0, 0, 0, 0, 0 ]]
        return mtx

        
def getConfMtxResult(pred, target, tol=0.5):
        mtx = genConfMtx()
        row = -1
        p = open(pred, 'r')
        t = open(target, 'r')
        if pred[0] == 'w':
                cutoff = 4898
        else:
                cutoff = 1599
        pred = []
        target = []
        for each in p:
                pred.append(float(each))
        for each in t:
                target.append(float(each))        
        for each in range(0,cutoff):
                row += 1
                # ignore R-datafile rows
                if (str(pred[each]).find('.') == -1):
                        continue
                # throw out any outliers
                if (target[each] < 4 or target[each] > 8):
                        continue
                if (pred[each] < 4-tol or pred[each] > 8+tol):
                        continue
                # row offset
                row = int(target[each] - 3)

                # col offset
                for i in range(4,9):
                        if ((pred[each] > i - tol) and (pred[each] < i + tol)):
                                col = i - 3

                mtx[row][col] += 1
        print mtx[0][0]
        print mtx[0]
        print mtx[1]
        print mtx[2]
        print mtx[3]
        print mtx[4]
        print mtx[5]

        return mtx

def predictKfold(train, traint, test, testt, runs, datafile):
        err = []
        cnt = 0
        pred = []
        tgts = []
        
        allpred = []
        alltgt = []
        # per datafile params
        if (datafile[0] == 'w'):
                rowcnt = '4898'
                outfn = 'ww-'
        else:
                rowcnt = '1599'        
                outfn = 'rw-'
        
        for res in range(0,runs):
                tgts.append(str(13) + '\n' + str(rowcnt) + str('\n'))
                pred.append(str(14) + '\n' + str(rowcnt) + str('\n'))
                for folds in range(0,5):                        
                        net = rbf.rbf(train[folds],traint[folds],22,0,0)
                        nbf = 2000
                        net.rbftrain(train[folds],traint[folds],0.00001,nbf)
                        output = net.rbffwd(test[folds])
                        err.append(average(abs(output - testt[folds])))                        
                        for e in output:
                                pred.append(str(e[0]) + str('\n'))
                                allpred.append(e[0])
                        for e in testt[folds]:
                                tgts.append(str(e[0])[0] + str('\n'))
                                alltgt.append (int(str(e[0])[0]))
        print(average(err))
        f = open(outfn + 'pred-' + str(average(err))[0:5] + '.csv','w')
        for each in pred:
                f.write(each)                
        f.close()
        f = open(outfn + 'tgts-' + str(average(err))[0:5] + '.csv','w')
        for each in tgts:
                f.write(each)                
        f.close()


def main():
        #findRbfs()
        #findParameters()

        datafile = 'red-headless.csv'
        #datafile = 'white-headless.csv'
        data = loadtxt(datafile,delimiter=';')

        #(train, traint, test, testt) = genKfold(data)
        #predictKfold(train, traint, test, testt, 20, datafile)

        predfile = 'ww-pred.csv'
        tgtfile = 'ww-tgts.csv'
        #predfile = 'rw-pred.csv'
        #tgtfile = 'rw-tgts.csv'
        T = 1.
        mtx = getConfMtxResult(predfile, tgtfile, T)
        pass

if __name__ == "__main__":
        main()


