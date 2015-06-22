from pylab import *
import argparse
import time
import csv

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Plotter for stats_file",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--statfile", type=str, required=False, help='stat_file')
    parser.add_argument("-pt", "--polltime", type=float, default=2.0, help='time to sleep between polls')
    parser.add_argument("-ne", "--numepochs", type=int, default=200, help='number of epochs to plot before stopping')
    args = parser.parse_args()

    ion() 

    with open(args.statfile) as statfile:
        csvreader = csv.reader(statfile)
        head = csvreader.next()
    colids = range(len(head)-1)
    currow = 0
    while currow< args.numepochs:
        plotdata = np.loadtxt(open(args.statfile),delimiter=',',skiprows=1)
        if plotdata is not None and plotdata.ndim > 1 and plotdata.shape[0]>2:
            currow = plotdata.shape[0]
            for c in colids:
                subplot(3,1,c+1)
                cla()
                title(head[c+1])
                plot(plotdata[:,c+1])
                xlim([0, args.numepochs])
                draw()
            show()
        time.sleep(args.polltime)
    ioff()
    show()
                

                












