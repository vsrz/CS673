
# Code from Chapter 4 of Machine Learning: An Algorithmic Perspective
# by Stephen Marsland (http://seat.massey.ac.nz/personal/s.r.marsland/MLBook.html)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008

from numpy import *

iris = loadtxt('winequality-red.csv',delimiter=';')
iris[:,:11] = iris[:,:11]-iris[:,:11].mean(axis=0)
imax = concatenate((iris.max(axis=0)*ones((1,12)),iris.min(axis=0)*ones((1,12))),axis=0).max(axis=0)
iris[:,:11] = iris[:,:11]/imax[:11]
#print iris[0:12,:]

#target = zeros((shape(iris)[0],2));
#indices = where(iris[:,4]==0) 
#target[indices,0] = 1
#indices = where(iris[:,4]==1)
#target[indices,1] = 1
#indices = where(iris[:,4]==2)
#target[indices,0] = 1
#target[indices,1] = 1

#target = zeros((shape(iris)[0],10));
#indices = where(iris[:,11]==1)
#target[indices,0] = iris[:,11]
#print target
#target = zeros((shape(iris)[0],1));
idx = 0
target=[]
for each in iris[:,11]:
	target.append(each)
	idx += 1
#for each in iris[:,11]:
#	target.append(idx, each)
#	idx += 1

#indices = where(iris[:,11]==2)
#target[indices,1] = 1
#indices = where(iris[:,11]==3)
#target[indices,2] = 1
#indices = where(iris[:,11]==4)
#target[indices,3] = 1
#indices = where(iris[:,11]==5)
#target[indices,4] = 1
#indices = where(iris[:,11]==6)
#target[indices,5] = 1
#indices = where(iris[:,11]==7)
#target[indices,6] = 1
#indices = where(iris[:,11]==8)
#target[indices,7] = 1
#indices = where(iris[:,11]==9)
#target[indices,8] = 1


order = range(shape(iris)[0])
random.shuffle(order)
iris = iris[order,:]
#target = target[order,:]

train = iris[::2,0:11]
traint = target[::2]
valid = iris[1::11,0:11]
validt = target[1::11]
#test = iris[3::11,0:11]
#testt = target[3::11]
test = iris[:,0:11]
testt = iris[:,0:11]
#print train.max(axis=0), train.min(axis=0)

import rbf
net = rbf.rbf(train,traint,12,1,1)

net.rbftrain(train,traint,0.25,2000)
#net.confmat(train,traint)
#net.confmat(test,testt)
