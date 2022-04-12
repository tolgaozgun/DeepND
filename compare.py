import torch
import numpy as np
import pandas as pd

def main():
    regions = ["PFC","MDCBC","V1C","SHA"]
    periods = ["1-3","2-4","3-5","4-6","5-7","6-8","7-9","8-10","9-11","10-12","11-13","12-14","13-15"]

    print("Reading predictions...")
    #predictions = pd.read_csv("/mnt/DeepND/v3/mt_with_identity_newidset_moepli_results/predict_id.csv", header=0)
    predictions = pd.read_csv("/mnt/DeepND/v3/asd_tolga_e01Exp01/predict_asd.csv", header=0)
    probabilities = predictions.values[:,2].tolist()
    print("Predictions read!")


    for region in regions:
        for period in periods:
            file = "e01_comparison/" + region + period + "Result.txt"
            fout = open(file, "w+")
            fout.write("Loading measures... (Pagerank) for " + region + period + "\n")
            print("Loading measures... (Pagerank) for " + region + period)
            pr = torch.load(region + period + "Pagerank.pt")
            pagerank = list(pr.values())
            fout.write("Loading measures... (Between) for " + region + period + "\n")
            print("Loading measures... (Between) for " + region + period)
            between = torch.load(region + period + "Between.pt").cuda().type(torch.Tensor).tolist()
            print("Loading measures... (Close) for " + region + period)
            fout.write("Loading measures... (Close) for " + region + period + "\n")
            close = torch.load(region + period + "Close.pt").cuda().type(torch.Tensor).tolist()
            print("Loading measures... (Degree)")
            fout.write("Loading measures... (Degree) for " + region + period + "\n")
            degree = torch.load(region + period + "Degree.pt").cuda().type(torch.Tensor).tolist()
            print("Measures loaded!")
            fout.write("Measures loaded!\n")
            b = np.corrcoef(x=between, y=probabilities)
            c = np.corrcoef(x=close, y=probabilities)
            d = np.corrcoef(x=degree, y=probabilities)
            p = np.corrcoef(x=pagerank, y=probabilities)
            fout.write("Pagerank:\n")
            fout.write(str(p) + "\n")
            fout.write("Betweenness:\n")
            fout.write(str(b) + "\n")
            fout.write("Closeness:\n")
            fout.write(str(c) + "\n")
            fout.write("Degree:\n")
            fout.write(str(d) + "\n")
            print("Pagerank:")
            print(p.tolist(), "\n")
            print("Betweenness:")
            print(b.tolist())
            print("Closeness:")
            print(c.tolist(), "\n")
            print("Degree:")
            print(d.tolist(), "\n")
            fout.close()



    #print( "%s & %s & %.4f & %.4f & %.4f \\\\" % ("PFC", "3-5", b.tolist(), c.tolist(), d.tolist()))
    #fout.write("%s & %s & %.4f & %.4f & %.4f \\\\" % ("PFC", "3-5", b, c, d))


if __name__ == "__main__":
    main()