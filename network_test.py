import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
import time

if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print("Invalid file name.")
    #     sys.exit()

    print("11.")
    temp = torch.load("PFC3-5Tensor.pt").type(torch.LongTensor).detach().numpy().reshape(-1,2)
    print("12.")
    temp = list(map(tuple, temp))
    print("13.")
    G = nx.from_edgelist(temp)
    #G.edge()
    print("14.")
    #nx.draw(G)
    print("15.")
    #plt.show()
    print("16.")
    # fout = open("network_results_" + str(time.time()), "w+")
    # fout.write(file)
    print("Write successful.")
