from graphviz import Digraph

graph = Digraph(comment='The Round Table')

graph.node('A', 'King Arthur')
graph.node('B', 'Sir Bedevere the Wise')
graph.node('L', 'Sir Lancelot the Brave')

graph.edges(['AB', 'AL'])
graph.edge('B', 'L', constraint='false')

#print(graph.source)

graph.render('visualisation.gv', view= True)
