# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 10:32:04 2019

@author: harsh
"""




from collections import deque
def bfs(graph, root): 
    visited, queue = set(),deque([root])   
    visited.add(root)
    while queue:
        print(queue)
        vertex = queue.popleft()
        print("neighbour_current",vertex)
        for neighbour in graph[vertex]: 
            print(visited)
            if neighbour not in visited: 
                visited.add(neighbour) 
                queue.append(neighbour) 
    print(queue)
if __name__ == '__main__':
    graph = {0: [1, 2], 1: [2], 2: [3], 3: [1,2]} 
    bfs(graph, 0)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
from collections import deque   
    
def dfs(graph,node):
    visited=set()
    
    queue=deque([node])
    while(queue):
        vertex=visited.add(queue.popleft())
        for neighbours in graph[vertex]:
            if neighbours not in visited:
                queue.append(neighbours)
        
    
    

    
    
if __name__ == '__main__':
    graph = {0: [1, 2], 1: [2], 2: [3], 3: [1,2]} 
    bfs(graph, 0)
    