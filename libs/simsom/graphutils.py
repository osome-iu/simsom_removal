"""
Graph-related functions, e.g.: function create synthetic bot network and shuffle networks
** remember link direction is following, opposite of info spread!
"""

import igraph as ig
import random
import string
import numpy as np
from copy import deepcopy
import warnings
from typing import Dict, List

# random.seed(42)
# np.random.seed(42)


def read_empirical_network(file: str) -> ig.Graph:
    """
    Read a network from file path.
    """
    try:
        net = ig.Graph.Read_GML(file)

        # prevent errors with duplicate attribs
        net = _delete_unused_attributes(
            net, desire_attribs=["label", "party", "misinfo"]
        )
    except Exception as e:
        print("Exception when reading network")
        print(e.args)
    return net


def random_walk_network(net_size: int, p=0.5, k_out=3) -> ig.Graph:
    """
    Create a network using a directed variant of the random-walk growth model
    https://journals.aps.org/pre/abstract/10.1103/PhysRevE.67.056104
    Inputs:
        - net_size (int): number of nodes in the desired network
        - k_out (int): average no. friends for each new node
        - p (float): probability for a new node to follow friends of a friend (models network clustering)
    """
    if net_size <= k_out + 1:  # if super small just return a clique
        return ig.Graph.Full(net_size, directed=True)

    graph = ig.Graph.Full(k_out, directed=True)

    for n in range(k_out, net_size):
        target = random.choice(graph.vs)
        friends = [target]
        n_random_friends = 0
        for _ in range(k_out - 1):  # bao: why kout-1 and not kout?
            if random.random() < p:
                n_random_friends += 1

        friends += random.sample(
            graph.successors(target), n_random_friends
        )  # return a list of vertex id(int)
        friends += random.sample(range(graph.vcount()), k_out - 1 - n_random_friends)

        graph.add_vertex(n)  # n becomes 'name' of vertex

        edges = [(n, f) for f in friends]

        graph.add_edges(edges)
    return graph


def add_user_activity_attribute(
    network: ig.Graph, activity_differential=True, alpha=2.85, xmin=0.2
) -> ig.Graph:
    """
    Add node attribute "postperday": different levels of activity for users.
    Parameters:
        - network (igraph.DiGraph): human network where nodes have attribute "uid" (str)
        - activity_differential (bool): if True, user activity follows a power law distribution. If False, uniform
        - alpha (float): scaling parameter, assuming user activity follows a power law distribution. Default value 2.85 is estimated from a 10-day sample of 10% Twitter posts
        - xmin (float): minimum activity level of a user, lower bound of power law
    Returns a network G (igraph.DiGraph); nodes have the following attributes: uid (str), postperday (float)
    """
    # inverse transform sampling to get user activity from power law distribution
    if activity_differential:
        if (alpha is None) or (xmin is None):
            print(
                "Alpha and/or xmin are not given to specify the power law distribution of user activity, using default (2.85 and 0.2, respectively)."
            )
            alpha = 2.85
            xmin = 0.2
        activities = []
        for _ in range(network.vcount()):
            r = random.uniform(0, 1)
            activity = xmin * r ** (-1 / (alpha - 1))
            activities.append(activity)
    else:
        print("Uniform activity (each agent does 1 action at each timestep)")
        activities = list(np.ones(network.vcount()))
    network.vs["postperday"] = activities

    return network


def create_human_bot_network(
    human_network: ig.Graph,
    targeting_criterion=None,
    verbose=False,
    beta=0.05,
    gamma=0.05,
) -> ig.Graph:
    """
    Creates a network of humans and bots
    Parameters:
        - human network (igraph.DiGraph): human network
        - targeting_criterion (str): bot targeting strategies. if None, random targeting
        - verbose (bool): if True, print different steps of network creation
        - beta (float): bots/humans ratio (specifies size of bot subnetwork)
        - gamma (float): probability that a human follows each bot (bot infiltration)
    Returns a network G (igraph.DiGraph); nodes have the following attributes: bot (bool), uid (str), party (float), misinfo (float)
    """
    # Create bot subnetwork
    n_humans = human_network.vcount()
    if verbose:
        print("Generating bot subnetwork...")
    n_bots = int(n_humans * beta)
    B = random_walk_network(n_bots)
    B.vs["bot"] = [1 for _ in range(B.vcount())]

    # merge and add feed
    # b: Retain human and bot ids - TODO: prob won't be needed later
    alphas = list(string.ascii_lowercase)
    B.vs["uid"] = [str(node.index) + random.choice(alphas) for node in B.vs]

    if verbose:
        print("Merging human and bot networks...")
    G = human_network.disjoint_union(B)
    G = _delete_unused_attributes(G, desire_attribs=["uid", "bot", "party", "misinfo"])

    assert G.vcount() == n_humans + n_bots
    # b:now nodes are reindex so we want to keep track of which ones are bots and which are humans
    humans = [n for n in G.vs if n["bot"] == False]
    bots = [n for n in G.vs if n["bot"] == True]

    # Make following links from authentic agents to bots
    if verbose:
        print("Humans following bots...")

    if targeting_criterion is not None:
        try:
            if targeting_criterion == "hubs":
                w = [G.degree(h, mode="in") for h in humans]
            elif targeting_criterion == "partisanship":
                w = [abs(float(h["party"])) for h in humans]
            elif targeting_criterion == "misinformation":
                w = [float(h["misinfo"]) for h in humans]
            elif targeting_criterion == "conservative":
                w = [1 if float(h["party"]) > 0 else 0 for h in humans]
            elif targeting_criterion == "liberal":
                w = [1 if float(h["party"]) < 0 else 0 for h in humans]
            else:
                raise ValueError("Unrecognized targeting_criterion passed to init_net")
            probs = [i / sum(w) for i in w]

        except Exception as e:
            if verbose:
                print(e)
            warnings.warn(
                "Unable to implement targeting criterion due to missing node attribute in empirical network."
                "Using default targeting criterion (random)."
            )
            targeting_criterion = None

    for b in bots:
        n_followers = 0
        for _ in humans:
            if random.random() < gamma:
                n_followers += 1
        if targeting_criterion is not None:
            # get a sample of followers weighted by probs (WITH replacement)
            followers = np.random.choice(humans, n_followers, replace=False, p=probs)
        else:
            followers = random.sample(humans, n_followers)  # without replacement

        follower_edges = [(f, b) for f in followers]
        G.add_edges(follower_edges)

    return G


def init_net(
    targeting_criterion=None,
    verbose=False,
    igraph_fpath=None,
    n_humans=1000,
    beta=0.05,
    gamma=0.05,
    activity_differential=False,
    alpha=None,
    xmin=None,
    quality_settings: Dict[str, List] = None,
) -> ig.Graph:
    """
    Creates a network as input to SimSom model
    Parameters:
        - targeting_criterion (str): bot targeting strategies; if None, random targeting
        - verbose (bool): if True, print different steps of network creation
        - igraph_fpath (str): file path of the empirical follower network. If None, create a synthetic human subnetwork
            nodes in human_network must have attribute "label", indicating account id on the platform, e.g., user_id for Twitter
        - n_humans (int): size of human subnetwork
        - beta (float): bots/humans ratio (specifies size of bot subnetwork). If 0, create a human-only network
        - gamma (float): probability that a human follows each bot (bot infiltration). If 0, create a human-only network
        - activity_differential (bool): if True, user activity follows a power law distribution
        - alpha (float): scaling parameter, assuming user activity follows a power law distribution
        - xmin (float): minimum activity level of a user, lower bound of the power law
        - quality_settings (dict): dictionary containing the parameters for the quality distribution and the classes of users.
            size is the list indicating the percentage of users who have a distribution in a given class (the sum of the percentages must be 100),
            quality_distr contains the values of the parameters alpa, beta (of the beta distribution) and the bounds within which the quality values
            must fall (e.g. the values fall between 0 and 0.3 in the first case and 0 and 1 in the second case)
    Returns a network G (igraph.DiGraph) with the following attributes: bot (bool), uid (str), party (float), misinfo (float), postperday (float)
    """
    # TODO: change the name convention of H, B and G (single char makes it hard to refactor if needed)
    # TODO: add another synthetic model (scale-free)

    # Create authentic agent subnetwork
    if igraph_fpath is None:
        if verbose:
            print("Generating human network...")
        H = random_walk_network(n_humans)
    else:
        if verbose:
            print("Reading human network...")
        H = read_empirical_network(igraph_fpath)

    # Add attributes
    H.vs["bot"] = [0 for _ in range(H.vcount())]
    # get uid from empirical network or create new index
    if "label" in H.vs.attributes():
        H.vs["uid"] = [str(node["label"]) for node in H.vs]
    else:
        H.vs["uid"] = [str(node.index) for node in H.vs]

    # if we do not pass any "quality_settings" params we set a default params
    if not quality_settings:
        quality_settings = {
            "attributes": ["a"],
            "size": [100],
            "qualitydistr": {"a": None},
        }

    if sum(quality_settings["sizes"]) != 100:
        raise ValueError(
            "Error in proportion of user classes, enter % values that added together make 100%"
        )
    if set(quality_settings["qualitydistr"].keys()) != set(
        quality_settings["attributes"]
    ):
        raise ValueError(
            "Error in quality distribution, 'qualitydistr' should have the same keys as 'attributes'"
        )
    # unpack "quality_settings" params
    user_class = []
    user_distribution = []

    for _ in H.vs:
        (uclass,) = random.choices(
            quality_settings["attributes"], weights=quality_settings["sizes"], k=1
        )
        udistr = quality_settings["qualitydistr"][uclass]
        user_class.append(uclass)
        user_distribution.append(str(udistr))

    # add attributes to nodes
    H.vs["class"] = user_class
    H.vs["qualitydistr"] = user_distribution

    # Create network
    if (beta != 0) and (gamma != 0):
        # bot-human network
        G = create_human_bot_network(
            human_network=H,
            targeting_criterion=targeting_criterion,
            verbose=verbose,
            beta=beta,
            gamma=gamma,
        )
    else:
        # human-only network
        G = H

    G = add_user_activity_attribute(
        network=G,
        activity_differential=activity_differential,
        alpha=alpha,
        xmin=xmin,
    )

    return G


def rewire_preserve_degree(og_graph, iterations=5):
    """
    Returns a rewired graph where degree distribution is preserved.
    Parameters:
        - graph (igraph.Graph object): the graph to shuffle
        - iterations: number of times to rewire to make sure community structure is destroyed
    """

    graph = deepcopy(og_graph)  # rewire is done in place so we want to make a deepcopy
    indeg, outdeg = graph.indegree(), graph.outdegree()

    graph.rewire(n=iterations * graph.ecount())
    assert indeg == graph.indegree()
    assert outdeg == graph.outdegree()
    print("Finished shuffling network (degree-preserving)!")

    return graph


def rewire_random(og_graph, probability=1):
    """
    Returns a randomly rewired graph.
    Parameters:
        - og_graph (igraph.Graph object): the graph to shuffle
        - probability (float): constant probability with which each endpoint of each edge is rewired
    """

    graph = deepcopy(og_graph)  # rewire is done in place so we want to make a deepcopy
    graph.rewire_edges(prob=probability, loops=False, multiple=False)

    print(
        f"Finished shuffling network (each edge's endpoints are rewired with probability {probability})!"
    )

    return graph


def _is_ingroup(graph: ig.Graph, edge, party=None) -> bool:
    """
    Check if an edge connects 2 nodes from the same community (party).
    Make sure that graph has a 'party' attribute s.t. -1<party<1
    For Nikolov et al. (2019) empirical follower network, every node belongs to a community:
    Conservative: node['party'] > 0, Liberal: node['party'] < 0
    Parameters:
        - party (str): {conservative, liberal}
    Outputs:
        - True if the edge is between 2 nodes in the same community (if specified)
        - else False
    """

    source_com = graph.vs[edge.source]["party"]
    target_com = graph.vs[edge.target]["party"]

    if float(source_com) * float(target_com) > 0:
        if party is not None:
            if (party == "conservative") and (float(source_com) > 0):
                return True
            if (party == "liberal") and (float(source_com) < 0):
                return True
            else:
                return False
        else:
            return True
    else:
        return False


def _rewire_subgraph_by_edges(
    graph, edge_idxs, iterations=5, prob=0.5, loops=False, multiple=False
):
    # Returns subgraphs spanned by partisan links
    # delete_vertices=True: vertices not incident on any of the specified edges will be deleted from the result
    og_subgraph = graph.subgraph_edges([e.index for e in edge_idxs])
    subgraph = deepcopy(og_subgraph)

    for iter in range(iterations):
        # Each endpoint of each edge of the graph will be rewired with a constant probability, given in the first argument.
        subgraph.rewire_edges(prob=prob, loops=loops, multiple=multiple)

    assert sorted([node["name"] for node in og_subgraph.vs]) == sorted(
        [node["name"] for node in subgraph.vs]
    )
    print("Finished rewiring subgraph!")
    return subgraph


def rewire_preserve_community(graph, iterations=5):
    """
    Returns a rewired graph where degree community structure is preserved.
    Inputs:
        - graph (igraph.Graph object): the graph to shuffle
        - iterations (int): number of times to rewire to make sure community structure is destroyed
    """
    graph.vs["name"] = [str(node["id"]) for node in graph.vs]
    conservative_edges = [
        e for e in graph.es if _is_ingroup(graph, e, party="conservative")
    ]
    liberal_edges = [e for e in graph.es if _is_ingroup(graph, e, party="liberal")]
    outgroup_edges = [e for e in graph.es if not _is_ingroup(graph, e)]

    assert (
        len(outgroup_edges) + len(conservative_edges) + len(liberal_edges)
        == graph.ecount()
    )

    print("Rewiring subgraphs...")
    # Ingroup edges should be rewired within a group, outgroup edges should be rewired between groups
    left_graph = _rewire_subgraph_by_edges(
        graph, liberal_edges, iterations=iterations, prob=0.5
    )
    right_graph = _rewire_subgraph_by_edges(
        graph, conservative_edges, iterations=iterations, prob=0.5
    )
    outgroup_graph = _rewire_subgraph_by_edges(
        graph, outgroup_edges, iterations=iterations, prob=0.5
    )

    # Create a new graph with rewired edges
    # Add vertices and edges by node *name* insted of *index* (because igraph continuously reindexes the nodes and edges)
    right_rewired = [
        (right_graph.vs[e.source]["name"], right_graph.vs[e.target]["name"])
        for e in right_graph.es
    ]
    left_rewired = [
        (left_graph.vs[e.source]["name"], left_graph.vs[e.target]["name"])
        for e in left_graph.es
    ]
    outgroup_rewired = [
        (outgroup_graph.vs[e.source]["name"], outgroup_graph.vs[e.target]["name"])
        for e in outgroup_graph.es
    ]

    # Make new graph
    print("Make new graph from rewired edges")
    rewired = ig.Graph(directed=True)
    rewired.add_vertices([node["name"] for node in graph.vs])
    for attribute in graph.vs.attributes():
        rewired.vs[attribute] = graph.vs[attribute]

    all_edges = right_rewired + left_rewired + outgroup_rewired
    assert len(all_edges) == graph.ecount()
    rewired.add_edges(all_edges)

    print("Finished shuffling network (community-preserving)!")

    return rewired


def _delete_unused_attributes(
    net: ig.Graph, desire_attribs=["uid", "party", "misinfo"]
) -> ig.Graph:
    # delete unused attribs or artifact of igraph to maintain consistency
    # NOTE: Commented out to work with synthetic networks (no user metadata, random targeting)
    # For experiments where bots get humans to follow non-randomly (via partisanship, misinformation, etc.), uncomment the following
    # network_attribs = net.vs.attributes()
    # if len(set(desire_attribs) & set(network_attribs)) < len(desire_attribs):
    #     raise ValueError(f"one of the desire attribs {desire_attribs} not in network")
    for attrib in net.vs.attributes():
        if attrib not in desire_attribs:
            del net.vs[attrib]
    return net
