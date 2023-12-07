import argparse
import os
import sys
import traceback
from graphCR.evaluation import al_famer

data_sets = [
    '../../test_data/DS2-MusicBrainz/threshold_0.35',
    '../../test_data/DS2-MusicBrainz/threshold_0.40',
    '../../test_data/DS2-MusicBrainz/threshold_0.45',
    # 'E:/data/DS-C/DS-C/DS-C0/SW_0.7',
    # 'E:/data/DS-C/DS-C/DS-C0/SW_0.3',
    # 'E:/data/DS-C/DS-C/DS-C0/SW_0.5',
    # 'E:/data/DS-C/DS-C/DS-C50/SW_0.3',
    # 'E:/data/DS-C/DS-C/DS-C50/SW_0.5',
    # 'E:/data/DS-C/DS-C/DS-C50/SW_0.7',
    # 'E:/data/DS-C/DS-C/DS-C100/SW_0.3',
    # 'E:/data/DS-C/DS-C/DS-C100/SW_0.5',
    # 'E:/data/DS-C/DS-C/DS-C100/SW_0.7',
]
# ['bootstrap','bootstrap_comb']
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='cluster repair')
    parser.add_argument('--data_file', '-d', type=str, default='E:/data/DS-C/DS-C/DS-C0/SW_0.3', help='data file')
    parser.add_argument('--save_dir', '-o', type=str, default='result_al.csv', help='result directory')
    parser.add_argument('--save_initial', '-o2', type=str, default='result_initial_al.csv', help='input dimension')
    args = parser.parse_args()
    is_edge_wise = True
    sel_strategies = ['bootstrap_comb']
    if args.data_file is not None:
        print(args.data_file)
        print(args.save_dir)
        for ds in data_sets:
            for i in range(3):
                for strat in sel_strategies:
                    for k in [20]:
                        for b in [1000, 1500, 2000]:
                            al_famer.evaluate(input_folder=ds, is_edge_wise=is_edge_wise, initial_training=k,
                                              increment_budget=k,
                                              total_budget=b, selection_strategy=strat, output=args.save_dir,
                                              output_2=args.save_initial, error_edge_ratio=0)

