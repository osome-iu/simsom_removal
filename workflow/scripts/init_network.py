""" Script to initialize Information network based on a .json configuration file """

import simsom.graphutils as graphutils
import simsom.utils as utils
import sys
import argparse
import json
import igraph
from simsom.network import (
    HumanBotNetwork,
    IllegalActivityNetwork,
    HarmfulActivityNetwork,
)

# def init_igraph(net_specs):
#     legal_specs = utils.remove_illegal_kwargs(net_specs, graphutils.init_net)
#     # print(legal_specs)
#     G = graphutils.init_net(**legal_specs)
#     return G


def main(args):
    parser = argparse.ArgumentParser(
        description="initialize info system graph from human empirical network",
    )

    parser.add_argument(
        "-i",
        "--infile",
        action="store",
        dest="infile",
        type=str,
        required=False,
        help="path to input .gml follower network file",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        action="store",
        dest="outfile",
        type=str,
        required=True,
        help="path to out .gml info system network file (with bots and humans)",
    )
    parser.add_argument(
        "-c",
        "--config",
        action="store",
        dest="config",
        type=str,
        required=True,
        help="path to all configs file",
    )
    parser.add_argument(
        "-m",
        "--mode",
        action="store",
        dest="mode",
        type=str,
        required=False,
        help="modeling scenario: {'illegal', 'harmful', 'manipulation'}",
    )

    args = parser.parse_args(args)
    infile = (
        args.infile
    )  # infile is a json containing list of {"beta": 0.0, "gamma": 0.0}
    outfile = args.outfile
    configfile = args.config
    mode = args.mode

    net_spec = json.load(open(configfile, "r"))
    if net_spec["igraph_fpath"] is not None:
        if infile is not None:
            print(
                f"! Infile specified in config file. Overwriting with infile argument:{infile}"
            )
            net_spec.update({"igraph_fpath": infile})
        else:
            print(
                f"! Infile not specified. Using infile from config file: {net_spec['igraph_fpath']}"
            )

    # print(net_spec)
    try:
        if mode == "illegal":
            net_spec = utils.remove_illegal_kwargs(
                net_spec, IllegalActivityNetwork.__init__
            )
            G = IllegalActivityNetwork(**net_spec).net
        if mode == "harmful":
            net_spec = utils.remove_illegal_kwargs(
                net_spec, HarmfulActivityNetwork.__init__
            )
            G = HarmfulActivityNetwork(**net_spec).net
        if mode == "manipulation":
            net_spec = utils.remove_illegal_kwargs(net_spec, HumanBotNetwork.__init__)
            G = HumanBotNetwork(**net_spec).net

        G.write_gml(outfile)

    # Write empty file if exception so smk don't complain
    except Exception as e:
        print(
            "Exception when making infosystem network. \n"
            "Likely due to sampling followers in targeting criteria."
        )
        print(e)

        G = igraph.Graph()
        G.write_gml(outfile)


if __name__ == "__main__":
    main(sys.argv[1:])
