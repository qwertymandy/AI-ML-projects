from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    result = []
    while queue:
        vertex = queue.popleft()
        if vertex not in visited:
            result.append(vertex)
            visited.add(vertex)
            queue.extend(n for n in graph[vertex] if n not in visited)
    return result

#graph = {
    #'A': ['B', 'C'],
    #'B': ['D', 'E'],
    #'C': ['F'],
    #'D': [],
    #'E': ['F'],
    #'F': []
#}

graph = {
    'A': ['B', 'C'],
    'B': ['G'],
    'C': ['D', 'E'],
    'D': [],
    'E': ['F'],
    'F': [],
    'G': []
}

# Run BFS starting from node 'A'
visited_nodes = bfs(graph, 'A')
print("visited nodes: ", visited_nodes)