import os
import pickle


EDGES = [
    ("webservice1", "mobservice1"),
    ("webservice1", "mobservice2"),
    ("webservice2", "mobservice1"),
    ("webservice2", "mobservice2"),
    ("webservice1", "redisservice1"),
    ("webservice1", "redisservice2"),
    ("webservice2", "redisservice1"),
    ("webservice2", "redisservice2"),
    ("mobservice1", "redisservice1"),
    ("mobservice1", "redisservice2"),
    ("mobservice2", "redisservice1"),
    ("mobservice2", "redisservice2"),
    ("logservice1", "dbservice1"),
    ("logservice1", "dbservice2"),
    ("logservice2", "dbservice1"),
    ("logservice2", "dbservice2"),
    ("logservice1", "redisservice1"),
    ("logservice1", "redisservice2"),
    ("logservice2", "redisservice1"),
    ("logservice2", "redisservice2"),
    ("dbservice1", "redisservice1"),
    ("dbservice1", "redisservice2"),
    ("dbservice2", "redisservice1"),
    ("dbservice2", "redisservice2"),
    ("logservice1", "logservice2"),
    ("logservice2", "logservice1"),
]

OUTPUT_DIR = "/GAIA_data/data/gaia/events"
NODES_PATH = os.path.join(OUTPUT_DIR, "nodes.pkl")
EDGES_PATH = os.path.join(OUTPUT_DIR, "edges.pkl")


def save_pickle(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def build_graph(edges):
    nodes = sorted({node for edge in edges for node in edge})
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}

    src_ids = [node_to_idx[src] for src, _ in edges]
    dst_ids = [node_to_idx[dst] for _, dst in edges]
    return nodes, (src_ids, dst_ids)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    nodes, indexed_edges = build_graph(EDGES)
    save_pickle(NODES_PATH, nodes)
    save_pickle(EDGES_PATH, indexed_edges)

    print(f"Saved {len(nodes)} nodes to {NODES_PATH}")
    print(f"Saved {len(EDGES)} edges to {EDGES_PATH}")
