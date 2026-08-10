from collections import defaultdict, deque
from queue import Queue
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
    graph_copy = assign_max_distances_from_start(
        graph=graph_copy,
        start_node=start_node,
        distance_key=distance_key,
    )
    nodes_groups = group_nodes_by_sorted_key_value(graph=graph_copy, key=distance_key)
    return [node for node_group in nodes_groups for node in node_group]


def assign_max_distances_from_start(
    graph: nx.DiGraph, start_node: str, distance_key: str = "distance"
) -> nx.DiGraph:
    nodes_to_consider = deque([start_node])
    # Use qsize() in original; equivalent with len(deque) and faster
    while nodes_to_consider:
        node_to_consider = nodes_to_consider.popleft()
        predecessors = list(graph.predecessors(node_to_consider))
        if not predecessors:
            distance_from_start = 0
        else:
            # All() can exit early, but in practice we'll almost always need to check all;
            # amalgamate the check and max in a single loop to avoid double iteration
            max_parent_distance = None
            for p in predecessors:
                parent_distance = graph.nodes[p].get(distance_key)
                if parent_distance is None:
                    break
                if max_parent_distance is None or parent_distance > max_parent_distance:
                    max_parent_distance = parent_distance
            else:
                distance_from_start = max_parent_distance + 1
                graph.nodes[node_to_consider][distance_key] = distance_from_start
                nodes_to_consider.extend(graph.successors(node_to_consider))
                continue
            # Not all predecessors have distance; skip to next node (do not enqueue children)
            continue
        graph.nodes[node_to_consider][distance_key] = distance_from_start
        nodes_to_consider.extend(graph.successors(node_to_consider))
    return graph


def group_nodes_by_sorted_key_value(
    graph: nx.DiGraph,
    key: str,
    excluded_nodes: Optional[Set[str]] = None,
) -> List[List[str]]:
    if excluded_nodes is None:
        excluded_nodes = set()
    key2nodes = defaultdict(list)
    for node_name, node_data in graph.nodes(data=True):
        if node_name in excluded_nodes:
            continue
        key2nodes[node_data[key]].append(node_name)
    sorted_key_values = sorted(list(key2nodes.keys()))
    return [key2nodes[d] for d in sorted_key_values]
