import argparse
from robots.sim_rccar.analysis.analyze_aggregate_percent import AnalyzeAggPerSimRCcar

############
### main ###
############

def main(args):
    analyze = AnalyzeAggPerSimRCcar()
    analyze.percent_plot(args.data_dir, args.save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir')
    parser.add_argument('--save_path')
    args = parser.parse_args()
    main(args)

