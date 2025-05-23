import os
import torch
from data.data_factory import data_provider
from models import EDformer
    

def stringify_setting(args, complete=False):
    if not complete:
        # first two conditions for specific ablations
        if args.task_name == 'classification' and args.seq_len != 48:
            return f"{args.data_path.split('.')[0]}_{args.model}_sl_{args.seq_len}"
        elif args.task_name == 'long_term_forecast' and args.seq_len != 96:
            return f"{args.data_path.split('.')[0]}_{args.model}_sl_{args.seq_len}"
        
        return f"{args.data_path.split('.')[0]}_{args.model}"
    
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}'.format(
        args.model,
        args.data_path.split('.')[0],
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads
    )
    
    return setting

dual_input_users = [
    'EDformer'
]

class Exp_Basic(object):
    model_dict = {
        'EDformer': EDformer
    }
    
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        
        self.setting = stringify_setting(args)
        
        if args.itr_no is not None:
            self.output_folder = os.path.join(
                args.result_path, self.setting, str(args.itr_no)
            )
        else:
            self.output_folder = os.path.join(args.result_path, self.setting)
            
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder, exist_ok=True)
        print(f'Experiments will be saved in {self.output_folder}')
        
        self.dataset_map = {}

    def _build_model(self):
        raise NotImplementedError
    
    def load_best_model(self):
        best_model_path = os.path.join(self.output_folder, 'checkpoint.pth')
        print(f'Loading model from {best_model_path}')
        self.model.load_state_dict(torch.load(best_model_path))

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self, flag='test'):
        if flag not in self.dataset_map:
            self.dataset_map[flag] = data_provider(self.args, flag)
            
        return self.dataset_map[flag] 

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
