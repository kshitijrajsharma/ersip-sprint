import numpy as np
import higra as hg


def attribute_standard_deviation(tree, vertex_weights, area=None, leaf_graph=None):
    """
    Compute standard deviation of vertex weights for each node in the tree.
    
    Parameters
    ----------
    tree : higra tree
        Component tree
    vertex_weights : ndarray
        Vertex weights (typically the input image)
    area : ndarray, optional
        Precomputed area attribute
    leaf_graph : higra graph, optional
        Leaf graph
        
    Returns
    -------
    ndarray
        Standard deviation for each node
    """
    if area is None:
        area = hg.attribute_area(tree, leaf_graph=leaf_graph)
    
    mean = hg.attribute_mean_vertex_weights(tree, vertex_weights, area=area, leaf_graph=leaf_graph)
    vertex_list = hg.attribute_vertex_list(tree)
    
    std = np.zeros(tree.num_vertices())
    vertex_weights_flat = vertex_weights.ravel()
    
    for node_idx in range(tree.num_vertices()):
        vertices = vertex_list[node_idx]
        if len(vertices) > 0:
            values = vertex_weights_flat[vertices]
            node_mean = mean[node_idx]
            variance = np.mean((values - node_mean) ** 2)
            std[node_idx] = np.sqrt(variance)
    
    return std