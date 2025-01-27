"""
Network classes to model different scenarios: illegal content spread, harmful content spread, bad actor manipulation
Definitions:
- both illegal content and harmful content to be messages that has quality=0. However, to model different spreading dynamics, different networks are used. 
- the term "bot" refers to users who consistently post zero-quality content and models different classes of users across different scenarios
    - In modeling bad actor manipulation (original SimSoM modeling), "bots" refer to trolls, social bots, cyborgs
    - In modeling illegal content spread, "bots" refer to users posting illegal content who need to be suspended 
- "humans" refer to normal users


Graph-related functions, e.g.: function create synthetic bot network and shuffle networks
** remember link direction is following, opposite of info spread!
"""

## UTILS

import igraph as ig
import random
import string
import numpy as np
from copy import deepcopy
import warnings

# random.seed(42)
# np.random.seed(42)


# TODO: Add typing to methods
# required network attributes for experiments on manipulation (human-bot network)
REQUIRED_ATTRIBS_HUMANBOT_NET = {
    "uid",
    "bot",
    "party",
    "misinfo",
    "postperday",
    "qualitydistr",
    "class",
}
# required network attributes for experiments on removal (network without bots)
REQUIRED_ATTRIBS_HUMAN_NET = {"uid", "postperday", "qualitydistr", "class"}


def read_empirical_network(file):
    """
    Read a network from file path.
    """
    try:
        raw_net = ig.Graph.Read_GML(file)

        # prevent errors with duplicate attribs
        net = _delete_unused_attributes(
            raw_net, desire_attribs=["label", "party", "misinfo"]
        )
    except Exception as e:
        print("Exception when reading network")
        print(e.args)
    return net


def _delete_unused_attributes(net, desire_attribs=["uid", "party", "misinfo"]):
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


def random_walk_network(net_size, p=0.5, k_out=3):
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


class Network:
    """
    Base class for a network. Every node has an attribute "uid" (str) and  "postperday" (float).
    Attributes:
        net (igraph.DiGraph): a network where nodes have the following attributes: uid (str), postperday (float).
    """

    def __init__(
        self,
        igraph_fpath,
        activity_differential=False,
        alpha=None,
        xmin=None,
        net_size=1000,
        p=0.5,
        k_out=3,
    ):
        """
        Read empirical network from file path or create a synthetic network using random-walk growth model

        Args:
            igraph_fpath (str): The file path to the igraph.
            activity_differential (bool): Flag to enable activity differential.
            alpha (float): The alpha value.
            xmin (float): The xmin value.
            net_size (int): The size of the network.
            p (float): The probability for a new node to follow friends of a friend.
            k_out (int): The average number of friends for each new node.
        Returns:
            None
        """

        self.igraph_fpath = igraph_fpath
        self.activity_differential = activity_differential
        self.alpha = alpha
        self.xmin = xmin
        self.net_size = net_size
        if self.igraph_fpath is not None:
            self.net = read_empirical_network(igraph_fpath)
        else:
            self.net = random_walk_network(net_size, p=p, k_out=k_out)

        # get uid from empirical network or create new index
        if "label" in self.net.vs.attributes():
            self.net.vs["uid"] = [str(node["label"]) for node in self.net.vs]
        else:
            self.net.vs["uid"] = [str(node.index) for node in self.net.vs]

        self.add_user_activity_attribute(activity_differential, alpha, xmin)

    def __repr__(self):
        return f"Network({self.igraph_fpath},{self.activity_differential},{self.alpha},{self.xmin},{self.net_size})"

    def __str__(self):
        return f"Base Network \n{self.net.summary()}"

    def set_network(self, igraph):
        """
        Set the Network net attribute to the given igraph.
        """
        self.net = igraph
        return None

    def add_node_attributes(self, attributes, attr_values):
        """
        Add attributes to the nodes of the network.

        Args:
            attributes (list of str): List of attribute names.
            attr_values (list of list): List of values corresponding to the attributes.

        Returns:
            None
        """

        for attribute, values in zip(attributes, attr_values):
            self.net.vs[attribute] = values
        return None

    def delete_node_attributes(self, attributes):
        """
        Delete attributes from the nodes of the network.

        Args:
            attributes (list of str): List of attribute names to be deleted.
        Returns:
            None
        """
        for attribute in self.net.vs.attributes():
            if attribute in attributes:
                del self.net.vs[attribute]
        return None

    def add_user_activity_attribute(
        self, activity_differential=True, alpha=2.85, xmin=0.2
    ):
        """
        Adds node attribute "postperday" to model different levels of activity for users.

        Args:
            network (igraph.DiGraph): Human network where nodes have attribute "uid" (str).
            activity_differential (bool): If True, user activity follows a power law distribution. If False, uniform.
            alpha (float): Scaling parameter, assuming user activity follows a power law distribution. Default value 2.85 is estimated from a 10-day sample of 10% Twitter posts.
            xmin (float): Minimum activity level of a user, lower bound of power law.

        Returns:
            None
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
            for _ in range(self.net.vcount()):
                r = random.uniform(0, 1)
                activity = xmin * r ** (-1 / (alpha - 1))
                activities.append(activity)
        else:
            print("Uniform activity (each agent does 1 action at each timestep)")
            activities = list(np.ones(self.net.vcount()))
        self.net.vs["postperday"] = activities

        return None

    def add_user_quality_attribute(self, quality_settings=None):
        """
        Adds node attribute "qualitydistr" to model different group of users.

        Args:
            quality_settings (dict): dictionary containing the parameters for the quality distribution and the classes of users.
            size is the list indicating the percentage of users who have a distribution in a given class (the sum of the percentages must be 100),
            quality_distr contains the values of the parameters alpa, beta (of the beta distribution) and the bounds within which the quality values
            must fall (e.g. the values fall between 0 and 0.3 in the first case and 0 and 1 in the second case)

        Returns:
            None
        """
        print("Adding user quality attribute...")
        # if we do not pass any "quality_settings" params we set a default params
        if not quality_settings:
            # in this case all users create messages with quality follows the exponential func (see `Message`)
            quality_settings = {
                "attributes": ["normal"],
                "sizes": [100],
                "qualitydistr": {"normal": None},
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

        for _ in self.net.vs:
            (uclass,) = random.choices(
                quality_settings["attributes"], weights=quality_settings["sizes"], k=1
            )
            udistr = quality_settings["qualitydistr"][uclass]
            user_class.append(uclass)
            user_distribution.append(str(udistr))

        # add attributes to nodes
        self.net.vs["class"] = user_class
        self.net.vs["qualitydistr"] = user_distribution

        return None


class NetworkFromIgraphObject(Network):
    def __init__(self, igraph):
        self.net = igraph


class HumanBotNetwork(Network):
    def __init__(
        self,
        igraph_fpath,
        activity_differential=False,
        alpha=None,
        xmin=None,
        net_size=1000,  # n_humans
        beta=0.05,
        gamma=0.05,
        targeting_criterion=None,
    ):
        """
        Network to model manipulation of social media consisting of 2 sub-networks: human subnetwork and bot subnetwork
        Bots are connected to human uniformly with probability gamma
        The bot subnetwork includes bad actors posting messages with quality=0 ; size: net_size * beta.
        The human subnetwork includes normal users posting high-quality messages most of the time ; siz: net_size.

        Every node has an attribute "bot" (bool) and "uid" (str).
        See paper: https://arxiv.org/pdf/1907.06130.pdf for details.

        Args:
            igraph_fpath (str): file path to the igraph file of the empirical human network.
            activity_differential (bool): whether users have different levels of activity.
            alpha (float): alpha value for activity differential.
            xmin (float): xmin value for activity differential.
            net_size (int): the size of the human subnetwork.
            beta (float): bots/humans ratio (specifies size of bot subnetwork)
            gamma (float): probability that a human follows each bot (bot infiltration)
            targeting_criterion (str): bot targeting strategies. if None, random targeting

        Returns:
            None
        """
        print("Generating human subnetwork...")
        super().__init__(igraph_fpath, activity_differential, alpha, xmin, net_size)
        self.human_subnetwork = Network(
            igraph_fpath, activity_differential, alpha, xmin, net_size
        )
        self.human_subnetwork.add_node_attributes(
            ["bot"], [[0] * self.human_subnetwork.net.vcount()]
        )
        self.beta = beta
        self.gamma = gamma
        self.targeting_criterion = targeting_criterion
        self.n_humans = self.human_subnetwork.net.vcount()
        self.n_bots = int(self.n_humans * beta)
        self.bot_ig_fpath = None

        print("Generating bot subnetwork...")
        self.bot_subnetwork = Network(
            self.bot_ig_fpath, activity_differential, alpha, xmin, self.n_bots
        )
        self.bot_subnetwork.add_node_attributes(
            ["bot"], [[1] * self.bot_subnetwork.net.vcount()]
        )
        self.add_node_attributes(["bot"], [[1] * self.bot_subnetwork.net.vcount()])

        print("Merging human and bot subnetworks...")
        self.set_network(
            self.merge_human_bot_network(
                self.human_subnetwork,
                self.bot_subnetwork,
                self.targeting_criterion,
                self.gamma,
            )
        )
        # add quality attribute
        user_class = ["bot" if node["bot"] else "normal" for node in self.net.vs]
        # beta distr of (0.01, 5, 0, 0) create messages with quality=0
        user_distribution = [
            (0.01, 5, 0, 0) if node["bot"] else None for node in self.net.vs
        ]

        self.add_node_attributes(["class"], user_class)
        self.add_node_attributes(["qualitydistr"], user_distribution)

        assert self.net.vcount() == self.n_humans + self.n_bots
        assert all(i in self.net.vs.attributes() for i in REQUIRED_ATTRIBS_HUMANBOT_NET)

    def __repr__(self):
        return f"HumanBotNetwork({self.igraph_fpath}, {self.activity_differential}, {self.alpha}, {self.xmin}, {self.net_size}, {self.beta}, {self.gamma}, {self.targeting_criterion})"

    def __str__(self):
        return f"HumanBotNetwork \n{self.net.summary()}"

    def merge_human_bot_network(
        self,
        human_network,
        bot_network,
        targeting_criterion=None,
        gamma=0.05,
    ):
        """
        Merges the human and bot subnetworks.

        Args:
            human_network (igraph.DiGraph): The human network.
            bot_network (igraph.DiGraph): The bot network.
            targeting_criterion (str): The bot targeting strategies. If None, random targeting.
            gamma (float): The probability that a human follows each bot (bot infiltration).

        Returns:
            igraph.DiGraph: A network G. Nodes have the following attributes: bot (bool), uid (str), party (float), misinfo (float).
        """
        alphas = list(string.ascii_lowercase)
        bot_network.add_node_attributes(
            ["uid"],
            [[str(node.index) + random.choice(alphas) for node in bot_network.net.vs]],
        )
        G = human_network.net.disjoint_union(bot_network.net)
        # delete_unused_attributes
        _delete_unused_attributes(G, desire_attribs=REQUIRED_ATTRIBS_HUMANBOT_NET)
        # b:now nodes are reindex so we want to keep track of which ones are bots and which are humans
        humans = [n for n in G.vs if n["bot"] == False]
        bots = [n for n in G.vs if n["bot"] == True]

        # Make following links from authentic agents to bots
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
                    raise ValueError(
                        "Unrecognized targeting_criterion passed to init_net"
                    )
                probs = [i / sum(w) for i in w]

            except Exception as e:
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
                followers = np.random.choice(
                    humans, n_followers, replace=False, p=probs
                )
            else:
                followers = random.sample(humans, n_followers)  # without replacement

            follower_edges = [(f, b) for f in followers]
            G.add_edges(follower_edges)

        return G


class IllegalActivityNetwork(Network):
    """
    Network to model illegal content spread: A sub-population of agents post illegal content consistently, the rest post illegal content sometimes randomly.
    """

    def __init__(
        self,
        igraph_fpath,
        activity_differential=False,
        alpha=None,
        xmin=None,
        net_size=1000,
        p=0.5,
        k_out=3,
        quality_settings={
            "attributes": ["illegal", "normal"],
            "sizes": [10, 90],
            "total_illegal_frac": 0.01,
            "qualitydistr": {"illegal": (3, 30, 0, 1), "normal": (0.1, 89.9, 0, 1)},
        },
    ):
        """
        Args:
            igraph_fpath (str): file path to the igraph file of the empirical human network.
            activity_differential (bool): whether users have different levels of activity.
            alpha (float): alpha value for activity differential.
            xmin (float): xmin value for activity differential.
            net_size (int): the size of the human subnetwork.
            quality_settings (dict): describing the fractions of the network more/less likely to post illegal content and distribution of quality for each group (beta distributions)
            Default values calibrated for Fb data (total illegal content: #SORs/MAARs = 0.2)
        Returns: None
        """
        super().__init__(
            igraph_fpath, activity_differential, alpha, xmin, net_size, p, k_out
        )
        self.quality_settings = quality_settings
        self.add_user_quality_attribute(self.quality_settings)
        _delete_unused_attributes(self.net, desire_attribs=REQUIRED_ATTRIBS_HUMAN_NET)
        # add dummy bot attribute to work with `simsom` original code
        self.net.vs["bot"] = [0] * self.net.vcount()
        assert all(i in self.net.vs.attributes() for i in REQUIRED_ATTRIBS_HUMAN_NET)

    def __repr__(self):
        return f"IllegalActivityNetwork({self.igraph_fpath}, {self.activity_differential}, {self.alpha}, {self.xmin}, {self.net_size}, {self.quality_settings})"

    def __str__(self):
        return f"IllegalActivityNetwork \n{self.net.summary()}"


class HarmfulActivityNetwork(Network):
    """
    Network to model harmful content spread: There are two group of users with different propensity to post low-quality information.
    The quality distribution of each group follows a beta distribution.

    Args:

    Returns: None
    """

    def __init__(
        self,
        igraph_fpath,
        activity_differential=False,
        alpha=None,
        xmin=None,
        net_size=1000,
        p=0.5,
        k_out=3,
        quality_settings={
            "attributes": ["harmful", "normal"],
            "sizes": [10, 90],
            "qualitydistr": {"harmful": (0.4, 2, 0, 0.3), "normal": (0.5, 0.15, 0, 1)},
        },
    ):
        """
        Args:
            igraph_fpath (str): file path to the igraph file of the empirical human network.
            activity_differential (bool): whether users have different levels of activity.
            alpha (float): alpha value for activity differential.
            xmin (float): xmin value for activity differential.
            net_size (int): the size of the human subnetwork.
            quality_settings (dict): describing the fractions of the network more/less likely to post illegal content and distribution of quality for each group (beta distributions)

        Returns: None
        """
        super().__init__(
            igraph_fpath, activity_differential, alpha, xmin, net_size, p, k_out
        )
        self.quality_settings = quality_settings
        self.add_user_quality_attribute(self.quality_settings)
        _delete_unused_attributes(self.net, desire_attribs=REQUIRED_ATTRIBS_HUMAN_NET)
        # add dummy bot attribute to work with `simsom` original code
        self.net.vs["bot"] = [0] * self.net.vcount()
        assert all(i in self.net.vs.attributes() for i in REQUIRED_ATTRIBS_HUMAN_NET)

    def __repr__(self):
        return f"HarmfulActivityNetwork({self.igraph_fpath}, {self.activity_differential}, {self.alpha}, {self.xmin}, {self.net_size}, {self.quality_settings})"

    def __str__(self):
        return f"HarmfulActivityNetwork \n{self.net.summary()}"
