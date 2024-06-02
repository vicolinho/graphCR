import argparse

from graphCR.evaluation import al_famer

data_sets = [
    #'../../test_data/DS1-Geographical/threshold_0.75'
    # '../../test_data/DS1-Geographical/threshold_0.8',
    # '../../test_data/DS1-Geographical/threshold_0.85',
    # '../../test_data/DS1-Geographical/threshold_0.90',
    # '../../test_data/DS1-Geographical/threshold_0.95',
    # '../../test_data/DS2-MusicBrainz/threshold_0.35',
    # '../../test_data/DS2-MusicBrainz/threshold_0.40',
    # '../../test_data/DS2-MusicBrainz/threshold_0.45',
    # 'E:/data/DS3-NCVR_5_Party/threshold_0.75/threshold_0.75'
    # ,'E:/data/DS3-NCVR_5_Party/threshold_0.80/threshold_0.80',
    # 'E:/data/DS3-NCVR_5_Party/threshold_0.85/threshold_0.85'
    # 'E:/data/DS-C/DS-C/DS-C0/SW_0.3',
    # 'E:/data/DS-C/DS-C/DS-C0/SW_0.5',
    'E:/data/DS-C/DS-C/DS-C0/SW_0.7',
    # 'E:/data/DS-C/DS-C/DS-C26/SW_0.3',
    # 'E:/data/DS-C/DS-C/DS-C26/SW_0.5',
    # 'E:/data/DS-C/DS-C/DS-C26/SW_0.7',
    # 'E:/data/DS-C/DS-C/DS-C32/SW_0.3',
    # 'E:/data/DS-C/DS-C/DS-C32/SW_0.5',
    # 'E:/data/DS-C/DS-C/DS-C32/SW_0.7',
    # 'E:/data/DS-C/DS-C/DS-C50/SW_0.3',
    # 'E:/data/DS-C/DS-C/DS-C50/SW_0.5',
    # 'E:/data/DS-C/DS-C/DS-C50/SW_0.7',
    # 'E:/data/DS-C/DS-C/DS-C62A/SW_0.3',
    # 'E:/data/DS-C/DS-C/DS-C62A/SW_0.5',
    # 'E:/data/DS-C/DS-C/DS-C62A/SW_0.7',
    # 'E:/data/DS-C/DS-C/DS-C80/SW_0.3',
    # 'E:/data/DS-C/DS-C/DS-C80/SW_0.5',
    # 'E:/data/DS-C/DS-C/DS-C80/SW_0.7',
    # 'E:/data/DS-C/DS-C/DS-C100/SW_0.3',
    # 'E:/data/DS-C/DS-C/DS-C100/SW_0.5',
    # 'E:/data/DS-C/DS-C/DS-C100/SW_0.7',
]
# ['info', 'info_opt', 'bootstrap','bootstrap_comb']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='cluster repair')
    parser.add_argument('--data_file', '-d', type=str, default='E:/data/DS-C/DS-C/DS-C0/SW_0.3', help='data file')
    parser.add_argument('--save_dir', '-o', type=str, default='result_al_gpt.csv', help='result directory')
    parser.add_argument('--save_initial', '-o2', type=str, default='result_initial_al.csv', help='input dimension')
    parser.add_argument('--api_key', '-ak', default='', type=str)
    parser.add_argument('--use_gpt', '-gpt', default=1, type=int, help='0=not used, 1=used')
    parser.add_argument('--model_name', '-n', default='gpt-3.5-turbo', type=str, help='model name')
    args = parser.parse_args()
    is_edge_wise = True

    DISTANCE = 'distance'
    DEGREE = 'degree'
    BETWEENNESS = 'betweenness'
    CLOSENESS = 'closeness'
    PAGE_RANK = 'pagerank'
    SIZE = 'size'
    COMPLETE_RATIO = 'complete_ratio'
    CLUSTER_COEFF = 'cluster_coefficient'
    FEATURE_VECTOR = 'h'
    UNA_RATIO = 'una_ratio'
    BRIDGES = 'bridges'
    # original features = ['pagerank', 'closeness', 'cluster_coefficient', 'betweenness']
    # ['pagerank', 'closeness', 'cluster_coefficient', 'betweenness', 'degree', 'size', 'una_ratio']
    features = [['pagerank', 'closeness', 'cluster_coefficient', 'betweenness']]
    sel_strategies = ['bootstrap_comb']
    dexter_considered_atts = {"famer_product_name"
                              "famer_model_list", "famer_model_no_list",
                              "famer_brand_list", "famer_keys", "<page title>", "optical zoom", "digital zoom",
                              "resolution",
                              "camera dimension"}

    if args.data_file is not None:
        print(args.data_file)
        print(args.save_dir)
        for ds in data_sets:
            for i in range(1):
                for strat in sel_strategies:
                    for k in [20]:
                        for b in [1000, 1500, 2000]:
                            for f in features:
                                al_famer.evaluate(input_folder=ds, features=f, is_edge_wise=is_edge_wise,
                                                  initial_training=k,
                                                  increment_budget=k,
                                                  total_budget=b, selection_strategy=strat, output=args.save_dir,
                                                  output_2=args.save_initial, error_edge_ratio=0,
                                                  use_gpt=args.use_gpt,
                                                  considered_atts=dexter_considered_atts,
                                                  api_key=args.api_key, model_name=args.model_name,
                                                  cache_path='requested_pairs.csv')

