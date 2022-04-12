import os
import sys
import pandas as pd
import numpy as np
import networkx as nx
import torch
import time
from numba import jit, cuda, errors, numba

def create_network_list(args):
    network_files = []
    names = []
    splitted_tokens = args.split(",")
    #test_network_path = "Data/" # main.py
    test_network_path = "/mnt/oguzhan/oguzhan_workspace/EdgeTensors/PointEight/" #NEO
    #test_network_path = "/media/hdd1/oguzhan/oguzhan_workspace/EdgeTensors/PointEight/" # GPU2
    regions = ["PFC", "MDCBC", "V1C", "SHA"]
    periods = ["1-3", "2-4", "3-5", "4-6", "5-7", "6-8", "7-9", "8-10", "9-11", "10-12", "11-13", "12-14", "13-15"]
    for token in splitted_tokens:
        if token.strip() == "brainspan_all":
            for region in regions:
                for period in periods:
                    network_files.append(test_network_path + region + period + "wTensor.pt")
                    names.append(region + period)
        elif token.strip() == "brainspan_no_overlap":
            for region in regions:
                for period in ["1-3", "4-6", "7-9", "10-12", "13-15"]:
                    network_files.append(test_network_path + region + period + "wTensor.pt")
                    names.append(region + period)
        else:
#            if token.strip().split("-")[0].isnumeric() and token.strip().split("-")[1].isnumeric():
#                regions = ["PFC", "MDCBC", "V1C", "SHA"]
#                for region in regions:
#                    network_files.append(test_network_path + region + token.strip() + "wTensor.pt")
#                    names.append(region + period)
            if "PFC" in token.strip() or "MDCBC" in token.strip() or "V1C" in token.strip() or "SHA" in token.strip():
                network_files.append(test_network_path + token.strip() + "wTensor.pt")
                names.append(token.strip())
            else:
                network_files.append(token.strip())
                names.append(token.strip())

    graph_list = []
    i = 1
    first = time.time()
    for network in network_files:
        start = time.time()
        print("loading network number", i, ",", names[i - 1])
        temp = torch.load(network).cuda().type(torch.LongTensor).detach().numpy().reshape(-1,2)
        temp = list(map(tuple, temp))
        G = nx.from_edgelist(temp)
        graph_list.append(G)
        print("Network", i, ",", names[i - 1], "loaded in", (time.time() - start), "seconds")
        i += 1

    print("All networks loaded in", (time.time() - first), "seconds")
    return graph_list, names


#@numba.jit#(target='cuda')
def centrality_measures(G, i, name):
    cent_start_time = time.time();
    last_time = time.time()
    # print("Calculating centrality measure - degree")
    #
    # degree = list(nx.degree_centrality(G).values())
    # print("Degree took ", (time.time() - last_time), " seconds")
    #
    # degree_time = time.time()
    # print("Saving degree measure")
    #
    # t_degree = torch.tensor(degree)
    # torch.save(t_degree, name + "Degree.pt")
    #
    # print("Degree measure saved in ", (time.time() - degree_time), " seconds")
    #
    #
    # last_time = time.time()
    # print("Calculating centrality measure - between")
    # between = list(nx.betweenness_centrality(G).values())
    # print("Between took ", (time.time() - last_time), " seconds")
    #
    # last_time = time.time()
    # print("Calculating centrality measure - close")
    # close = list(nx.closeness_centrality(G).values())
    # print("Close took ", (time.time() - last_time), " seconds")
    print("Calculating centrality measure - pagerank")
    pagerank = nx.pagerank(G)

    print("Calculated network number ", i, " named ", name, " in ", (time.time() - last_time), " seconds, ",
          (time.time() - cent_start_time), " total.")


    print("Saving pagerank measure")
    pr_time = time.time()
    #storch.tensor(pagerank)
    torch.save(pagerank, name + "Pagerank.pt")
    print("Pagerank measure saved in ", (time.time() - pr_time), " seconds")

    # between_time = time.time()
    # print("Saving betweenness measure")
    #
    # t_between = torch.tensor(between)
    # torch.save(t_between, name + "Between.pt")
    #
    # print("Betweenness measure saved in ", (time.time() - between_time), " seconds")
    #
    # close_time = time.time()
    # print("Saving Closure measure")
    #
    # t_close = torch.tensor(close)
    # torch.save(t_close, name + "Close.pt")
    #
    # print("Closure measure saved in ", (time.time() - close_time), " seconds, total of ", (time.time() - degree_time),
    #       " seconds elapsed saving.")


def main():
    start_time = time.time()

    # Create and open the log file in write mode.c
    fout = open("npy_analyze_res" + str(time.time()), "w+")


    # Parse the config file.
    print("Parsing config file.")
    fout.write("Parsing config file.")

   # os.environ["NUMBA_BOUNDSCHECK"] = str(0)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"

    filepath = 'analyzer.cfg'
    with open(filepath) as fp:
        for cnt, line in enumerate(fp):
            if '#' not in line and line.strip() != "":  # Ignore lines with a '#' character and blank lines.
                splitted_line = line.split("=")

                if len(splitted_line) != 2:
                    print("Invalid parameter format at line ", cnt + 1, ", using default value for this parameter.")
                    continue

                parameter_name = splitted_line[0].strip()
                parameter_value = splitted_line[1].strip()

                if parameter_name == "networks":
                    network_selector = parameter_value

                elif parameter_name == "system_gpu_mask":
                    system_gpu_mask = parameter_value
                    #from utils import *

                elif parameter_name == "calculate_measures":
                    calcMeasures = int(parameter_value)

    config_time = (time.time() - start_time)
    print("Config file has been parsed in %s seconds, %s total" % (config_time, (time.time() - start_time)))
    fout.write("Config file has been parsed in %s seconds, %s total" % (config_time, (time.time() - start_time)))

    net_start_time = time.time()
    print("Loading networks: %s" % network_selector)
    fout.write("Loading networks: %s" % network_selector)
    networks, names = create_network_list(network_selector)
    net_total_time = time.time() - net_start_time
    print("Loaded networks in %s seconds, %s total" % (net_total_time, (time.time() - start_time)))
    fout.write("Loaded networks in %s seconds, %s total" % (net_total_time, (time.time() - start_time)))

    cent_start_time = time.time()
    print("Loading centrality measures")
    fout.write("Loading centrality measures")

    if calcMeasures == 1:
        i = 1
        for G in networks:
            name = names[i - 1]
            centrality_measures(G, i, name)
            i += 1

    cent_total_time = time.time() - cent_start_time
    print("Loaded measures in %s seconds, %s total" % (cent_total_time, (time.time() - start_time)))
    fout.write("Loaded measures in %s seconds, %s total" % (cent_total_time, (time.time() - start_time)))


if __name__ == "__main__":
    main()