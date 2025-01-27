"""

2024-08-16
Use to run SimSomV1: model before adding algorithm variations. Use to run exps with smaller time delay values and robustness results

Default values and range of parameter swipe
All agents in the network are the same and have equal probability of spreading illegal content. 

"""

import numpy as np

follower_network = "follower_network.gml"
mode = "igraph"

##### DEFAULT VALUES #####
## diffusion/network
DEFAULT_MU = 0.5
DEFAULT_SIGMA = 15
DEFAULT_ACTIVITY_DIFF = True
## bad actors
DEFAULT_PHI = 0
DEFAULT_THETA = 1
DEFAULT_BETA = 0
DEFAULT_GAMMA = 0
DEFAULT_STRATEGY = None

## simulation
DEFAULT_RHO = 0.9
DEFAULT_EPSILON = 0.0001
## moderation
DEFAULT_ILLEGAL_CONTENT_PROBABILITY = 0.1
DEFAULT_MODERATE = False
DEFAULT_HALFLIFE = 1

##### NETWORK INITIALIZATION (WITH BOTS) #####
ACTIVITY_DIFFERENTIAL = [True, False]
TARGETING = [None, "hubs", "partisanship", "conservative", "liberal", "misinformation"]
GAMMA = sorted(list(10.0 ** (np.arange(-4, 0))))

##### EXPLORE OTHER BOT PARAMS #####
THETA_SWIPE = [1, 2, 4, 8, 16, 32, 64]
PHI_SWIPE = list(np.arange(0, 1.1, 0.1))

##### EXPLORE MODERATION PARAMS #####s
# HALFLIFE = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
HALFLIFE = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 0.5, 0.25, 0.125, 0.0625, 0.03125]

ILLEGAL_NETWORK_SWIPE = [
    {
        "attributes": ["illegal", "normal"],
        "sizes": [10, 90],
        "total_illegal_frac": 0.0001,
        "qualitydistr": {"illegal": (3, 5997.0, 0, 1), "normal": (0.1, 1799.9, 0, 1)},
    },
    {
        "attributes": ["illegal", "normal"],
        "sizes": [10, 90],
        "total_illegal_frac": 0.001,
        "qualitydistr": {"illegal": (3, 597.0, 0, 1), "normal": (0.1, 179.9, 0, 1)},
    },
    {
        "attributes": ["illegal", "normal"],
        "sizes": [10, 90],
        "total_illegal_frac": 0.01,
        "qualitydistr": {"illegal": (3, 30, 0, 1), "normal": (0.1, 89.9, 0, 1)},
    },
]

# NOTE: Convergence by illegal frac since that's the quantity of interest
infosys_default = {
    "verbose": False,
    "output_cascades": False,
    "epsilon": DEFAULT_EPSILON,
    "rho": DEFAULT_RHO,
    "mu": DEFAULT_MU,
    "sigma": DEFAULT_SIGMA,
    "phi": DEFAULT_PHI,
    "theta": DEFAULT_THETA,
    "moderate": DEFAULT_MODERATE,
    "moderation_half_life": DEFAULT_HALFLIFE,
    "modeling_legality": True,
    "converge_by": "illegal_frac",
}

# Total illegal content fraction=0.1
default_illegal_net = {
    "igraph_fpath": follower_network,
    "activity_differential": DEFAULT_ACTIVITY_DIFF,
    "quality_settings": {
        "attributes": ["illegal", "normal"],
        "sizes": [10, 90],
        "total_illegal_frac": 0.01,
        "qualitydistr": {"illegal": (3, 30, 0, 1), "normal": (0.1, 89.9, 0, 1)},
    },
}
