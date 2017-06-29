import argparse
from robots.rccar.algorithm.analyze_aggregate_percent import AnalyzeAggPerRCcar

############
### main ###
############

def main(args):
    analyze = AnalyzeAggPerRCcar()
    analyze.percent_plot(args.data_dir, args.save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir')
    parser.add_argument('--save_path')
    args = parser.parse_args()
    main(args)

