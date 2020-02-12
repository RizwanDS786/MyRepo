# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 19:30:13 2019

@author: harsh
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 18:57:17 2019

@author: harsh
"""


#any prime number can be reprseneted in the form of 6K+1 or 6k-1
#This principle can be used to find out if the number is prime or not
#Check until number//2
def prime_number(number):
    if(number==1):
        return "neither prime nor even"
    if(number==2 or number==3):
        return "prime"
    for i in range(2,(number//2)+1):
        print(number)
        if(number%i==0):
            print("num is not prime")
            return "not a prime"
            break
        else:
            return "prime"
prime_number(4)

5//2

def count_all_primes(number):
    primes=[1]*number
    primes[0]=0
    primes[1]=0
    for i in range(2,len(primes)):
        primes[i*i:number:i]=[0]*()
            
    
    
""" __
y=\/x

thershold=10^-4


y^2=x

2y(dy)=dx

y=  dx
    --
    dy*2

assume initial square root=1

and move it towards the derivative



"""   
    

def squareroot(number):
    threshold=0.001
    initial=1
    

squareroot(2)
        





