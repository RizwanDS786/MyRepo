# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 17:05:33 2019

@author: harsh
"""

#Creating a linked list
#create nodes for each node
#Point each node next to next nodes value

class Node:
    def __init__(self,val,next=None):
        self.val=val
        self.next=None 


class Solution:
    def traverse(self,node=None):
        while(node!=None):
            print(node.val)
            node=node.next
    
    def find_mid_node(self,node):
        #Traverse entire linked list and find the length
        #Find the mid of the element         
        length=0
        count=0
        current=node
        while(current!=None):
            #print(node.val)
            length=length+1
            print(current.val)
            current=current.next
        mid=(length+1)//2
        print(mid)
        
        current=node
        while(count<mid):
            count=count+1
            current=current.next
        return current.val
    
    def reverse(self,node):
        #Reverese the linked list
        #Create a pointer to None
        #Store a node into temp and point it to the prev
        #Store the next node into prev
        prev=None
        print(node.val)
        while node:
            temp=node
            node=node.next
            temp.next=prev
            prev=temp
        print(prev.val)
        return prev
    
    def couple_reverse(self,node):
        while(node):
            temp = node        
            node1 = node.next
            node1.next = temp
            temp.next = node1
            node = node1.next
            return self.couple_reverse(node)

        
            
#3 --> 4 --> 5--->6--7---8-9
        
#4--3--6--5--8--7-9
        

#  prev= None

# temp =node (3--> 4-->5)
# node= node.next(4-->5) 
        
# temp.next=node

node1=Node(4)
node2=Node(5)
node3=Node(6)
node4=Node(7)
node1.next=node2
node2.next=node3
node3.next=node4


solution=Solution()
#solution.traverse(node1)
#solution.find_mid_node(node1)
solution.reverse(node4)
solution.couple_reverse(node1)
print(node1.next.next.next.val)
print(node1.next.next.next.next.val)



#Create a cyclic linked list and see if there is a cycle in it
class Node:
    def __init__(self,val,next=None):
        self.val=val
        self.next=None

class Cycle:
    def cyclicity(self,node1):
        try:
            pointer1=node1
            pointer2=node1.next
            while(pointer1!=pointer2):
                pointer1=pointer1.next
                pointer2=pointer2.next.next
            return True
        except:
            return False
                     
node1=Node(5)
node2=Node(10)
node3=Node(8)
node4=Node(13)
node5=Node(12)
node1.next=node2
node2.next=node3
node3.next=node4
node4.next=node5
node5.next=node3.next
cycle=Cycle()
cycle.cyclicity(node1)



array=[4,-3,-1,-6,7,9,0,-1,-2]
for i in range(1,len(array)):
    array[i]=max(array[i],array[i]+array[i-1])

print(max(array))


max_substring=None
mx_length=0

#sum(array)

maximum = 0
for i in range(0,len(array)):
    sum_array = sum(array[0:i+1])
    if(sum_array < array[i]):
        maximum = array[i]
    else:
        maximum = sum_array
print(maximum)    


graph = { "a" : ["c"],
          "b" : ["c", "e"],
          "c" : ["a", "b", "d", "e"],
          "d" : ["c"],
          "e" : ["c", "b"],
          "f" : []
        } 



def generate_edges(graph):
    edges = []

    # for each node in graph
    for node in graph:

        # for each neighbour node of a single node
        for neighbour in graph[node]:
            # if edge exists then append
            edges.append((node, neighbour))
    return edges


    
#visted=()
#queue=[0]




