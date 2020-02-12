# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 15:17:11 2019

@author: harsh
"""

#Know about queues and stacks for binary tree and Graph iteravtive

import queue
L=queue.deque()
for i in range(5):
    L.append(i)
    
print(L)
#Pops fifo
print(L.popleft())
#Pops LIFO
L.pop()


class Node:
    def __init__(self,data):
        self.data=data
        self.left=None
        self.right=None
        
class Solution:
    def preorderTraversal(self,root):
        stack=[root]
        print(stack)
        res=[]
        while(stack):
            node=stack.pop()
            print(res)
            if(node!=None):
                res.append(node.data)
                stack.append(node.left)
                stack.append(node.right)
                print(res)
                print("iteration")
        return res

root=Node(10)
root.left=Node(6)
root.right=Node(5)
root.left.left=Node(9)
root.left.right=Node(7)
root.right.right=Node(2)
root.right.left=Node(3)

solution=Solution()
solution.preorderTraversal(root)


""""

    10--nlr
   / \
   6  5
  /\  /\
 9 7  2 3
/
13


""""





class Node:
    def __init__(self,val,left=None,right=None):
        self.val=val
        self.left=left
        self.right=right
        
class Order:
    def preordertraversal(self,node):
        "nlr-- Visit node- then its left- then right (When you moved to left repeat the same process"
        stack=[node]
        res=[]
        while(stack):
            root=stack.pop()
            print(root)
            if(root):
                res.append(root.val)
                print(res)
                stack.append(root.right)
                stack.append(root.left)
                
    def preorder_recursion(self,node):
        if node:
            print(node.val)
            self.preorder_recursion(node.left)
            self.preorder_recursion(node.right)
        
    def postordertraversal(self,node):
        pass
    
    def postorder_recursive(self,node):
        if node:
            self.postorder_recursive(node.left)
            self.postorder_recursive(node.right)
            print(node.val)
            
    
    def inordertraversal(self,node):
        pass
    
    
    def inorderrecursive(self,node):
        if node:
            self.inorderrecursive(node.left)
            print(node.val)
            self.inorderrecursive(node.right)
    
    def levelordertraversal(self,node):
        q=Q
        
        
        

        
    
node=Node(10)
node.left=Node(6)
node.right=Node(5)
node.left.left=Node(9)
node.right.right=Node(3)
node.right.left=Node(2)
node.left.right=Node(7)
node.left.left.left=Node(13)
order=Order()
#order.preordertraversal(node)
#order.preorder_recursion(node)
#order.postorder_recursive(node)
#order.inorderrecursive(node)


