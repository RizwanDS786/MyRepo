# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 16:47:12 2019

@author: harsh
"""

#Program to reverse the string o(n)
def reverseString(s1):
    print(len(s1))
    reverse=""
    for i in  range(0,len(s1)):
        reverse=s1[i]+reverse
    print(reverse)
reverseString('Harsa')


#Duplicates alphabets of string:
def duplicates(s2):
    d={}
    count=0
    for i in range(1,len(s2)):
        if s2[i] not in d:
            d.update({s2[i]:count})
        else:
            d.update({s2[i]:count+1})
    print(d)
    for key,value in d.items():
        print(d.values())
            
duplicates('Harsha')


#Anagrams:
def anagrams(s1,s2):
    count=0
    count1=0
    d1={}
    d2={}
    for i in range(0,len(s1)):
        if i not in d1:
            d1.update({s1[i]:count})
        else:
            d1    
    for i in range(0,len(s2)):
        if i not in d2:
            d2.update({s2[i]:count1})
        else:
            d2.update({s2[i]:count1+1}) 
    print(d1,d2)
    if(d1==d2):
        print('Anagrams');
    else:
        print('No anagrams')
    
anagrams("aba","baaa")


#count of letters in words:
def letters(s1):
    count=0
    d1={}
    for i in range(0,len(s1)):
        if i not in d1:
            d1.update({s1[i]:count})
        else:
            d1.update({s1[i]:count+1}) 
    print(d1)
    for key,value in d1.items():
        print(d1.values())
letters("Harshavardhan")



s={'a':1}
print(s.get(s,0))




def reverseString(s):
    start=0
    end=len(s)-1
    while(start<end):
        s[start],s[end]=s[end],s[start]
        start=start+1
        end=end-1
    print(s)
        
reverseString(['h','v','a'])