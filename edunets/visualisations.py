# Inspired by https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/micrograd/micrograd_lecture_first_half_roughly.ipynb
import numpy as np
from graphviz import Digraph

def trace(root):
# builds a set of all nodes and edges in a graph
	nodes, edges = [], []
	def build(v):
		if v._key not in [n._key for n in nodes]:
			nodes.append(v)
			for child in v._children:
				edges.append((child, v))
				build(child)
	build(root)
	return nodes, edges

def readable_data(data):
	nan_data = np.isnan(data).sum() == data.flatten().shape[0]
	if len(data.shape) == 1 and data.shape[0] == 1:
		return str(data)
	return f"tensor{data.shape} = [nan]" if nan_data else f"tensor{data.shape}"

def draw_dot(root):
	dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right

	nodes, edges = trace(root)
	for n in nodes:
		uid = str(id(n))
		# for any value in the graph, create a rectangular ('record') node for it
		# if no gradient was required, simply show the data

		grad = n._grad
		label = n.label if n.label is not None else readable_data(n.data)

		if grad is not None:
			if isinstance(grad, np.ndarray):
				if np.isnan(grad).sum() == grad.flatten().shape[0]:
					grad = f"grad{grad.shape} = [nan]"
				else:
					grad = f"grad{grad.shape}"
			else:
				grad = str(grad)
		else:
			grad = "grad()"

		if n.requires_grad:
			dot.node(name = uid, 
					label = f"{label} | {grad}",
					shape='record')
		else:
			dot.node(name = uid, 
					label = label,
					shape='record')

		if n._op:
			dot.node(name=uid + n._op, label=n._op)
			dot.edge(uid + n._op, uid)

	for n1, n2 in edges:
		# connect n1 to the op node of n2
		dot.edge(str(id(n1)), str(id(n2)) + n2._op)

	return dot