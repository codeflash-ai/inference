from collections import defaultdict
from queue import Queue, SimpleQueue
from typing import List, Optional, Set

import networkx as nx
from networkx import DiGraph


def traverse_graph_ensuring_parents_are_reached_first(
    graph: DiGraph,
    start_node: str,
) -> List[str]:
    """
    This function works under assumption of common super-input node in the graph - otherwise,
    there is no common entry point to put as `start_node`.
    """
    graph_copy = graph.copy()
    distance_key = "distance"
    assign_max_distances_from_start(
        graph=graph_copy,
        start_node=start_node,
        distance_key=distance_key,
    )
    nodes_groups = group_nodes_by_sorted_key_value(graph=graph_copy, key=distance_key)
    return [node for node_group in nodes_groups for node in node_group]


def assign_max_distances_from_start(
    graph: nx.DiGraph, start_node: str, distance_key: str = "distance"
) -> nx.DiGraph:
    # Use SimpleQueue for better performance on single-threaded code (faster than Queue)
    nodes_to_consider = SimpleQueue()
    nodes_to_consider.put(start_node)
    # Cache the nodes' distance attribute to avoid repeated dictionary lookups
    nodes_distance = graph.nodes
    while not nodes_to_consider.empty():
        node_to_consider = nodes_to_consider.get()
        predecessors = list(graph.predecessors(node_to_consider))
        # Use short-circuiting and local lookups for speed
        if not all(
            nodes_distance[p].get(distance_key) is not None for p in predecessors
        ):
            continue
        if not predecessors:
            distance_from_start = 0
        else:
            # Efficiently compute max using generator expression
            distance_from_start = (
                max(nodes_distance[p][distance_key] for p in predecessors) + 1
            )
        nodes_distance[node_to_consider][distance_key] = distance_from_start
        for neighbour in graph.successors(node_to_consider):
            nodes_to_consider.put(neighbour)
    return graph


def group_nodes_by_sorted_key_value(
    graph: nx.DiGraph,
    key: str,
    excluded_nodes: Optional[Set[str]] = None,
) -> List[List[str]]:
    if excluded_nodes is None:
        excluded_nodes = set()
    key2nodes = defaultdict(list)
    # Avoid repeated conversion for excluded_nodes by using set directly and eliminating unnecessary list conversion
    for node_name, node_data in graph.nodes(data=True):
        if node_name in excluded_nodes:
            continue
        key2nodes[node_data[key]].append(node_name)
    # Use sorted with generator instead of explicit list conversion
    sorted_key_values = sorted(key2nodes.keys())
    return [key2nodes[d] for d in sorted_key_values]
