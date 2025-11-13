# This script performs a depth-first search (DFS) on a graph represented as an adjacency list.
def dfs(graph, start, visited=None, result=None):
    if visited is None:
        visited = set()
    if result is None:
        result = []
    visited.add(start)
    result.append(start)
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited, result)
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
visited_nodes = dfs(graph, 'A')
print("visited nodes: ", visited_nodes)