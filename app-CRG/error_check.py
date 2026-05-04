import sys
import json
import traceback
import re
import numpy as np
import pandas as pd
import networkx as nx
import ipaddress

class MyChecker():
    def __init__(self, ret_graph=None, ret_list=None):
        # No usar truthiness: un grafo vacío es falsy pero válido para chequear tipos.
        self.graph = ret_graph if ret_graph is not None else None
        self.output_list = ret_list if ret_list is not None else None

    def evaluate_all(self):
        if self.graph is not None:
            graph_checks = [self.verify_node_type]
            for check in graph_checks:
                try:
                    check()
                except Exception as e:
                    print("Check failed:", e)
                    print(traceback.format_exc())
                    return False, e
            return True, ""

        if self.output_list is not None:
            return True, ""

        # Caso defensivo: no hay nada que validar
        return False, ValueError("No output to verify (graph and output_list are None).")

    def verify_node_type(self):
        """
        Verify if each node's 'type' is one of the four allowed types.

        Args:
            graph (networkx.Graph): The graph to verify.

        Returns:
            bool: True if all types are valid, False otherwise.
        """
        allowed_types = set(['virtualmachines', 'Networkinterfaces', 'virtualnetworks', 'networksecuritygroups'])

        for node in self.graph.nodes():
            node_type = self.graph.nodes[node].get('type')
            if node_type:
                if node_type not in allowed_types:
                    print(f"Invalid type at node: {node} with type: {node_type}")
                    return False
        return True



