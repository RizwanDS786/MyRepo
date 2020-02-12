# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 02:29:42 2019

@author: harsh
"""
# reverse a array
def reverese(A,start,end):
    while (start<end):
        A[start],A[end]=A[end],A[start]
        start+=1
        end-=1
    print(A)
A=[1,2,7,4,5]
reverese(A,0,len(A)-1)

#Maximum of an array
def maxNum(A):
    maximum=A[0]
    for i in range(0,len(A)):
        print(i)
        if A[i]>maximum:
            maximum=A[i]
    return maximum
    

A=[1,2,7,4,5]
maxNum(A)        
reverese(A,0,4,2)
#print(A)



a=[1,3,4,5,6,7,3]
b={}
for i in a:
    if i not in b:
        b[i]=1
    else:
        b[i]+=1
for k,v in b.items():
    if(b.values()>2):
        print(b.keys())