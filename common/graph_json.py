"""Compatibilidad JSON node-link de NetworkX (clave ``links`` frente a ``edges``)."""
from __future__ import annotations

from typing import Any


def node_link_graph_compat(data: dict[str, Any]):
    """Construye un grafo desde un dict node-link.

    Los exports de ``nx.node_link_data`` usan ``links``; en NetworkX 3.x el
    lector por defecto busca ``edges``.
    """
    from networkx.readwrite import json_graph

    if isinstance(data, dict) and "links" in data:
        return json_graph.node_link_graph(data, edges="links")
    return json_graph.node_link_graph(data)
