# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 21:19:00 2019

@author: harsh
"""

#Dictionaries

"""Implement an algorithm to determine if a string has all unique characters"""

def unique_characters():
    string="123"
    d={}
    count=1
    for i in range(0,len(string)):
        if string[i] not in d:
            d.update({string[i]:count})
            return True
        else:
            return False
unique_characters()

"""What if you cannot use additional data structures?"""


a=[1,2,3]
b=[]

for i in range(0,len(a)+1):
    for j in range(1,len(a)+1):
        b.append(a[i:j])
for i in range(0,len(b)):
    print(sum(b[i]))



"""Given an array of integers, return indices of the two numbers such that they add up 
to a specific target."""

def two_sum_brute_force():
    a=[2,5,7,9]
    target=9
    for i in range(0,len(a)):
        for j in range(i+1,len(a)):
            if(a[i]+a[j]==target):
                return True
#Works but still a bug needs to be closed which is failing for [3,3] case
        
def two_sum_hash_approach():
    a=[2,5,7,9]
    target=9
    d={}
    for i in range(0,len(a)):
        if(a[i] not in d):
            d.update({d[a[i]]:(target-a[i])})
    print(d)
    
two_sum_hash_approach()
    
    



        
        

        