import argparse, os, torch, json, random, pickle
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_classification import Exp_Classification
from exp.exp_basic import *
import numpy as np

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True

def initial_setup(args):
    args.use_gpu = True if torch.cuda.is_available() and (not args.use_cpu) else False
    
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
        
    args.enc_in = args.dec_in = args.c_out = args.n_features
    set_random_seed(args.seed)

def main(args):
    initial_setup(args)

    print('Args in experiment:')
    print(args)
    
    if args.task_name == 'classification': Exp = Exp_Classification
    else: Exp = Exp_Long_Term_Forecast
    
    parent_seed = args.seed
    np.random.seed(parent_seed)
    experiment_seeds = np.random.randint(1e3, size=args.itrs)
    original_itr = args.itr_no
        
    for itr_no in range(1, args.itrs+1):
        if (original_itr is not None) and original_itr != itr_no: continue
        
        args.seed = experiment_seeds[itr_no-1]
        print(f'\n>>>> itr_no: {itr_no}, seed: {args.seed} <<<<<<')
        set_random_seed(args.seed)
        args.itr_no = itr_no
        
        exp = Exp(args)  # set experiments
        if args.train:
            print('>>>>>>> training : >>>>>>>>>')
            exp.train()

            exp.test(load_model=False, flag='val')
            print('>>>>>>> testing : <<<<<<<<<<<')
            exp.test(load_model=False, flag='test')
        else:
            print('>>>>>>> testing : <<<<<<<<<<<<')
            exp.test(load_model=True, flag=args.flag)
        print()
    
    args.seed = parent_seed
    config_filepath = os.path.join(args.result_path, stringify_setting(args), 'config.json')
    args.seeds = [int(seed) for seed in experiment_seeds]
    with open(config_filepath, 'w') as output_file:
        json.dump(vars(args), output_file, indent=4)
    
    torch.cuda.empty_cache()

def get_parser():
    parser = argparse.ArgumentParser(
        description='Run Timeseries Models', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast', 
        choices=['long_term_forecast', 'classification'], help='task name')
    parser.add_argument('--train', action='store_true', help='whether to train the model or test')
    parser.add_argument('--dry_run', action='store_true', help='runs for a single batch')
    parser.add_argument('--model', type=str, required=True, default='Transformer',
        choices=list(Exp_Basic.model_dict.keys()), help='model name')
    parser.add_argument('--seed', default=2024, help='random seed')
    parser.add_argument('--itrs', type=int, default=3, help='experiment repetition time from 1 to itrs')
    parser.add_argument('--itr_no', type=int, default=None, help='experiments number among itrs. 1<= itr_no <= itrs .')

    # data loader
    parser.add_argument('--data', type=str, default='custom', help='dataset type')
    parser.add_argument('--result_path', type=str, default='./results', help='root result output folder')
    parser.add_argument('--root_path', type=str, default='./dataset/electricity/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='electricity.csv', help='data file')
    parser.add_argument('--flag', type=str, default='test', choices=['train', 'val', 'test'],
        help='data split type')
    parser.add_argument('--features', type=str, default='MS', choices=['M', 'S', 'MS'],
                        help='forecasting task; M: multivariate predict multivariate, S: univariate predict univariate, MS: multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument(
        '--freq', type=str, default='h', choices=['s', 't', 'h', 'd', 'b', 'w', 'm'],
        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    # model define
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    
    parser.add_argument('--n_features', type=int, default=1, help='number of input fetures.')
    parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=256, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=3, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--conv_kernel', default=[18, 12], nargs="+", type=int,
        help='convolution kernel size list for MICN. Can be [seq_len/2, pred_len].')
    parser.add_argument('--seg_len', type=int, default=24,
                        help='the length of segmen-wise iteration of SegRNN')

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='optimizer learning rate')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_cpu', action='store_true', help='run on cpu. Uses GPU by default')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')
    
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
