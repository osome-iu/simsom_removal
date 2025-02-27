""" 
Create the empirical follower network as input to simulations.
    1. Reconstruct the follower network for the data files provided in doi.org/10.7910/DVN/6CZHH5
    2. Further filter to make the network more manageable: k-core decomposition and edge filtering

Date: Jun 10, 2024
Author: Bao Truong

"""

import networkx as nx
import json
import pandas as pd
import random
import os


def make_network(path, files):
    """
    Make directed network follower -> friend
    Get a subgraph of partisan users
    """

    stats = pd.read_csv(os.path.join(path, files["user_info"]), sep="\t")
    stats = stats.astype({"ID": str}).dropna(axis=0).drop_duplicates()

    with open(os.path.join(path, files["adjlist"])) as fp:
        adjlist = json.load(fp)
    # Convert all node names from int to str. Keys are already str
    adjlist = {k: [str(n) for n in vlist] for k, vlist in adjlist.items()}

    friends = stats[stats["ID"].isin(adjlist.keys())]
    nodes = friends["ID"].values
    print("Nodes that have partisanship info: ", len(nodes))
    user_dict = friends.to_dict(orient="records")
    user_dict = {
        user["ID"]: {
            "Partisanship": user["Partisanship"],
            "Misinformation": user["Misinformation"],
        }
        for user in user_dict
    }

    G = nx.DiGraph()
    # Directed network follower -> friend
    for s in nodes:
        G.add_node(
            s,
            partisanship=user_dict[s]["Partisanship"],
            misinfo=user_dict[s]["Misinformation"],
        )
        for f in adjlist[s]:
            G.add_edge(s, f)

    return filter_graph(G, nodes)


def filter_graph(G, nodes_to_filter):
    """
    - Reduce the size of network by retaining a k-core for k=94 (approx. 10k nodes in core)
    - Reduce density by deleting a random sample of edges:
        a randomly sampled number E of edges such that the average in-degree and out-degree are the same as in the original network
        (E = <k>N, where <k> is the average in/out-degree and N is the number of nodes).
    """

    average_friends = G.number_of_edges() / G.number_of_nodes()
    # Basic stats
    print(
        f"{G.number_of_nodes()} nodes and {G.number_of_edges()} edges initially"
        f"(average number of friends: {average_friends})"
    )

    friends = nx.subgraph(G, nodes_to_filter)
    print(
        f"{friends.number_of_nodes()} nodes and {friends.number_of_edges()} edges after filtering"
    )

    # k-core decomposition until ~ 10k nodes in core
    core_number = nx.core_number(friends)
    nodes = friends.number_of_nodes()
    k = 0
    while nodes > 10000:
        k_core = nx.k_core(friends, k, core_number)
        nodes = k_core.number_of_nodes()
        k += 10
    while nodes < 10000:
        k_core = nx.k_core(friends, k, core_number)
        nodes = k_core.number_of_nodes()
        k -= 1
    print(
        f"{k}-core has {k_core.number_of_nodes()} nodes, {k_core.number_of_edges()} edges"
    )

    # the network is super dense, so let us delete a random sample of edges
    # we can set the initial average in/out-degree (average_friends) as a target
    friends_core = k_core.copy()
    edges_to_keep = int(friends_core.number_of_nodes() * average_friends)
    edges_to_delete = friends_core.number_of_edges() - edges_to_keep
    deleted_edges = random.sample(friends_core.edges(), edges_to_delete)
    friends_core.remove_edges_from(deleted_edges)
    print(
        f"{k}-core after edge-sampling has {friends_core.number_of_nodes()} nodes, {friends_core.number_of_edges()} edges,"
        f" and average number of friends {friends_core.number_of_edges() / friends_core.number_of_nodes()}"
    )
    return friends_core


if __name__ == "__main__":
    ABS_PATH = "data"
    DATA_PATH = os.path.join(ABS_PATH, "follower_network")

    files = {
        # File has 3 columns: ID \t partisanship \t misinformation \n
        "user_info": "measures.tab",
        "adjlist": "anonymized-friends.json",
    }
    friends_core = make_network(DATA_PATH, files)
    nx.write_gml(friends_core, os.path.join(DATA_PATH, "follower_network.gml"))
