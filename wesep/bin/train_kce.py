# original
import argparse
import os
import copy
import yaml
import heapq

import torch
import torch.nn
import torch.distributed as dist
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from wesep.dataset.kce_dataset import build_kce_dataset as build_dataset
from wesep.utils.kce_utils import WarmUpLR, read_list, Recorder
from wesep.utils.yamlinclude import YamlIncludeConstructor

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        required=True,
        help="config file in yaml format e.g. config/ref.yaml"
    )
    parser.add_argument(
        '--step',
        required=False,
        default=0,
        type=int,
        help="step"
    )
    parser.add_argument(
        '--world_size',
        required=True,
        type=int,
        help="world size required by torch.distributed usually set it as 4"
    )
    parser.add_argument(
        '--rank',
        required=True,
        type=int,
        help="rank"
    )
    parser.add_argument(
        '--port',
        required=False,
        default="1234",
        type=str,
        help="port"
    )
    parser.add_argument(
        '--gpu',
        required=True,
        help='gpu id we used, we are not support train with cpu'
    )
    parser.add_argument(
        '--seed',
        default=2022,
        help="random seed"
    )
    parser.add_argument(
        '--exp_root',
        default='exp/',
        help="experinments root dir"
    )

    args = parser.parse_args()
    return args


class Trainer():
    def __init__(
        self,
        model_arch: str,
        config_file: dict,
        rank: int,
        world_size: int,
        random_seed=2022,
        #args: argparse.Namespace
    ):
        # init config info
        self.config_file = config_file
        self.data_config = config_file['data_config']
        self.exp_config = config_file['exp_config']


        self.best_ckpt_heap = []  # max-heap based on loss
        self.max_best_ckpt_to_keep = self.exp_config.get('max_best_ckpt_to_keep', 20)

        # continue training from break point
        if self.data_config['start_epoch'] != 0:
            config_file = '{}/model.yaml'.format(self.exp_config['exp_dir'])
            model_config = yaml.load(open(config_file), Loader=yaml.FullLoader)
            self.model_config = model_config
        else:
            self.model_config = config_file['model_config']
        self.model = model_arch(**self.model_config)
        self.rank = rank
        self.seed = random_seed
        self.device = torch.device('cuda')
        self.world_size = world_size
        # if (not os.path.isdir(self.exp_config['exp_dir'])) and (self.rank == 0):
        try:
            os.makedirs(self.exp_config['exp_dir'], exist_ok=True)
        except:
            raise FileNotFoundError("can not create exp dir: {}".format(self.exp_config['exp_dir']))


        # init recorder        
        self.exp_config['log_config']['filename'] = "{}/train.{}.log".format(
            self.exp_config['exp_dir'],
            self.rank
        )
        self.recorder = Recorder(self.exp_config) 

    def backup_configs(self):
        for k, v in self.config_file.items():
            self.recorder.info("{} config : {}".format(k.upper(), v))
        
        if (self.rank == 0) and (self.data_config['start_epoch'] == 0): 
            # model config backup in expdir
            mf = open("{}/model.yaml".format(self.exp_config['exp_dir']), 'w')
            yaml.dump(self.model_config, mf)
            # data config backup in expdir
            df = open("{}/data.yaml".format(self.exp_config['exp_dir']), 'w')
            yaml.dump(self.data_config, df)
            # exp config backup in expdir
            ef = open("{}/exp.yaml".format(self.exp_config['exp_dir']), 'w')
            yaml.dump(self.exp_config, ef)

    def init_data_loaders(self):
        num_workers = self.data_config.get('num_workers', 10)

        train_list_file = self.data_config['train_list']
        valid_list_file = self.data_config.get('valid_list', None)
        if valid_list_file:
            valid_list = read_list(valid_list_file, split_cv=False, shuffle=True)
            train_list = read_list(train_list_file, split_cv=False, shuffle=True)
        else:
            train_list, valid_list = read_list(train_list_file, split_cv=True, shuffle=True)

        if self.data_config.get('egs_format', False):
            egs_path = os.path.dirname(train_list_file)
            assert os.path.isfile("{}/train.samples".format(egs_path))
            with open("{}/train.samples".format(egs_path),'r') as ef:
                self.num_samples = int(ef.readline().strip())
        else:
            self.num_samples = len(train_list)
    
        self.train_dataset = build_dataset(
            copy.deepcopy(self.data_config),
            train_list,
            tag_of_dataset='train'
        )
        self.valid_dataset = build_dataset(
            copy.deepcopy(self.data_config),
            valid_list,
            tag_of_dataset='valid'
        )

        self.tr_loader = DataLoader(
            self.train_dataset,
            batch_size=None,
            num_workers=num_workers
        )
        self.cv_loader = DataLoader(
            self.valid_dataset,
            batch_size=None,
            num_workers=3
        )

        if self.data_config['start_epoch'] == 0:
            self.recorder.info(
                "Num Training samples: {}, Num Valid samples: {}".format(len(train_list), len(valid_list))
            )
            self.recorder.info(
                "Num Worker: {}".format(num_workers)
            )

    def init_opt_model(self):
        # warm_up setting: compute update steps per epoch
        self.batch_size = self.data_config['batch_size']
        world_size = self.world_size
        steps_per_epoch = self.num_samples // (self.batch_size * world_size)
        steps_per_epoch = 1 if steps_per_epoch == 0 else steps_per_epoch
        if self.exp_config.get('warm_up_peak_epoch', False):
            warm_up_peak_epoch = self.exp_config.get('warm_up_peak_epoch')
        else:
            warm_up_peak_epoch = 5
        warm_up_peak_step = warm_up_peak_epoch * steps_per_epoch

        # init model & optim & dist
        num_param = sum([v.numel() for v in self.model.parameters()])
        self.clip_value = self.exp_config.get('clip_value', 10.0)
        self.optim = optim.Adam(
            self.model.parameters(), 
            **self.exp_config['optim_config']
        )

        start_epoch = self.data_config.get('start_epoch', 0)
        if start_epoch != 0:
            ckpt = self.load_endpoint(self.data_config['start_epoch']-1)
            self.global_step = self.load_ckpt(ckpt)
        else:
            self.global_step = 0
        self.scheduler = WarmUpLR(self.optim, warmup_steps=warm_up_peak_step)
        self.scheduler.set_step(self.global_step)
        if self.exp_config.get('finetune', False):
            finetune_config = self.exp_config.get('finetune')
            #trained_ckpt = finetune_config['trained_ckpt']
            self.init_from_trained(**finetune_config)

        # init distributed training
        if self.world_size > 1:
            dist.init_process_group(
                'nccl', #TODO: check why dragon03 can not use nccl
                world_size=self.world_size,
                rank=self.rank
            )
            self.model.cuda()
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                find_unused_parameters=True,
            )
        else:
            self.model.to(self.device)

        # save init checkpoint
        if self.rank == 0:
            d_model = copy.deepcopy(self.model)
            if isinstance(d_model, torch.nn.parallel.DistributedDataParallel):
                d_model = d_model.module.state_dict()
            else:
                d_model = d_model.state_dict()

            self.recorder.save_state(
                epoch='init',
                model=d_model,
            )

        if start_epoch == 0:
            self.recorder.info("WarmUp epoch: {} WarmUp steps:{}".format(warm_up_peak_epoch, warm_up_peak_step))
            self.recorder.info("Gradient Clip Value: {}".format(self.clip_value))
            self.recorder.info("Number parameter: {}".format(num_param))
        else:
            self.recorder.info("Continue training from epoch: {}".format(start_epoch))

    # percentage_fix_layer: mean how many layer in """NOTE: TRAINED CKPT""" will be fixed
    # NOTE: THAT the percentage means layer in trained ckpt not in new ckpt
    def init_from_trained(self, trained_ckpt, percentage_fix_layer='last'):
        self.recorder.info(
                "Init from trained checkpoint {} {}\% of them will be fixed".format(trained_ckpt, percentage_fix_layer)
        )
        trained_ckpt = torch.load(trained_ckpt, map_location='cpu')
        trained_model = trained_ckpt['model']
        trained_keys = list(trained_model.keys())
        fix_component = []
        current_state_dict = self.model.state_dict()
        num_trained_layers = len(trained_keys)
        if isinstance(percentage_fix_layer, int) and (percentage_fix_layer > 1):
            percentage_fix_layer = float(percentage_fix_layer) / 100
        num_fix_layer = num_trained_layers - 2 if percentage_fix_layer == 'last' else float(percentage_fix_layer) * num_trained_layers
        self.recorder.info("load parameters from trained model: {}".format(trained_ckpt.keys()))
        for i, k in enumerate(trained_keys):
            traiend_param = trained_model[k]
            if (current_state_dict[k].size() == traiend_param.size()) and (i < num_fix_layer):
                fix_component.append(k)
            else:
                trained_model.pop(k)
        self.model.load_state_dict(trained_model, strict=False)
        for i, (name, param) in enumerate(self.model.named_parameters()):
            if name in fix_component:
                param.requires_grad = False
            else:
                self.recorder.info("{} are trainable".format(name))
        
    def load_ckpt(self, ckpt):
        ckpt_dict = torch.load(ckpt, map_location='cpu')
        model = ckpt_dict['model']
        opt = ckpt_dict['opt']
        step = ckpt_dict['step']

        self.optim.load_state_dict(opt)
        for state in self.optim.state.values():
            for k, v in state.items():
                if k == 'step':
                    continue
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
        self.model.load_state_dict(model)
        return step
    
    @torch.no_grad()
    def cross_valid(self):
        cv_model = copy.deepcopy(self.model)
        cv_model.eval()
        cv_detail_loss = {'total_loss': 0}
        num_utt = 0
        for batch_id, cv_data in enumerate(self.cv_loader):
            n = cv_data[0].size(0)
            cv_data = (d.to(self.device) for d in cv_data)
            num_utt += n
            total_loss, detail_loss = cv_model(cv_data) 
            # if 'sv_ce_loss' in detail_loss:
            #     detail_loss.pop('sv_ce_loss')   # meaningless for cv or adjusting lr

            detail_loss = self.detach_from_graph(detail_loss)
            cv_detail_loss['total_loss'] += (self.detach_from_graph(total_loss) * n)
            #detail_loss['total_loss'] = total_loss*n

            for key, value in detail_loss.items():
                if key not in cv_detail_loss.keys():
                    cv_detail_loss[key] = value*n
                else:
                    cv_detail_loss[key] += (value*n)

        cv_detail_loss = {
            key : value/num_utt for key, value in cv_detail_loss.items()
        }
        return cv_detail_loss
    
    def detach_from_graph(self, para):
        if isinstance(para, torch.Tensor):
            para = para.detach().clone()
        if isinstance(para, dict):
            para = {k: v.detach().clone() for k,v in para.items()}
        return para

    def detach_state_dict(self):
        d_model = copy.deepcopy(self.model)
        opt = copy.deepcopy(self.optim)
        if isinstance(d_model, torch.nn.parallel.DistributedDataParallel):
            d_model = d_model.module.state_dict()
        else:
            d_model = d_model.state_dict()
        opt = opt.state_dict()
        return d_model, opt
    
    def record_step(self, r_loss):
        assert (isinstance(r_loss, dict))
        for key, value in r_loss.items():
            assert(key in ['train', 'cv'])
            self.recorder.record_detail(
                value, self.epoch, self.global_step,
                model=None, opt=None, tag=key
            )

    def load_endpoint(self, epoch):
        exp_dir = self.exp_config['exp_dir']
        checkpoint_name = self.exp_config['checkpoint_name']
        ckpt = "{}/{}_{}.pt".format(exp_dir, checkpoint_name, epoch)
        if os.path.isfile(ckpt):
            return ckpt
        else:
            raise FileNotFoundError(
                "{} does not exits check the start epoch from yaml config file".format(ckpt)
            )
    
    def avg_model(self):
        # average the last 10 model
        max_epoch = self.data_config['epoch']
        avg_epoch = self.exp_config.get('avg_epoch', 10)
        #max_epoch = 50
        min_epoch = max_epoch - avg_epoch
        valid_ckpt = {k:0 for k in range(min_epoch, max_epoch)}
        valid_loss = []
        for e in range(min_epoch, max_epoch):
            ckpt = "{}/{}_{}.pt".format(self.exp_config['exp_dir'], self.exp_config['checkpoint_name'],e)
            ckpt = torch.load(ckpt, map_location='cpu')
            one_valid_loss = ckpt['cv_loss']['total_loss'].item()
            valid_ckpt[e] = ckpt['model']
            valid_loss.append(one_valid_loss)
        sort_idx = sorted(range(len(valid_loss)), key=lambda k: valid_loss[k])
        min_idx = sort_idx[:10]
        state = None
        avg_model = None
        for k in min_idx:
            k = k + min_epoch
            state = valid_ckpt[k]
            if avg_model == None:
                avg_model = state
            else:
                for k in avg_model.keys():
                    avg_model[k] += state[k]
        for k in avg_model.keys():
            if avg_model[k] is not None:
                avg_model[k] = torch.true_divide(avg_model[k], 10)
        self.recorder.save_state(avg_model, epoch='avg')

    def record_epoch(self, cv_loss=None, record_by_loss='total_loss'):
        model, opt = self.detach_state_dict()
        epoch = self.epoch
        ckpt_path = f"{self.exp_config['exp_dir']}/{self.exp_config['checkpoint_name']}_{epoch}.pt"
        
        self.recorder.record_epoch(
            self.epoch, self.global_step,
            model, opt, cv_loss
        )
        
        # 保留最优的 K 个 checkpoint（按 cv_loss 排序）
        if cv_loss is not None and f'{record_by_loss}' in cv_loss:
            loss = cv_loss[f'{record_by_loss}'].item() if isinstance(cv_loss[f'{record_by_loss}'], torch.Tensor) else float(cv_loss[f'{record_by_loss}'])

            # Python 默认是最小堆，所以取负数模拟最大堆（因为我们要保留“最小损失”的前10个）
            heapq.heappush(self.best_ckpt_heap, (-loss, ckpt_path))
            if len(self.best_ckpt_heap) > self.max_best_ckpt_to_keep:
                worst_ckpt = heapq.heappop(self.best_ckpt_heap)
                try:
                    os.remove(worst_ckpt[1])
                    self.recorder.info(f"Removed worst checkpoint during training: {worst_ckpt[1]}")
                except Exception as e:
                    self.recorder.info(f"Failed to remove {worst_ckpt[1]}: {e}")

    def print_best_checkpoints(self):
        self.recorder.info("=== Best Checkpoints Retained ===")
        for loss_neg, path in sorted(self.best_ckpt_heap, reverse=True):
            self.recorder.info(f"{path} | val_loss: {-loss_neg:.4f}")

    # def average_model(self):
    #     for loss_neg, path in sorted(self.best_ckpt_heap, reverse=True):
    #         self.recorder.info(f"{path} | val_loss: {-loss_neg:.4f}")

    def train(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(args.seed)
        self.model.to(self.device)
        self.model.train()
        start_epoch = self.data_config['start_epoch']
        end_epoch = self.data_config['epoch']
        self.recorder.info("Start training the log is written in {}".format(
                self.exp_config['log_config']['filename']
            )
        )

        #TODO: add model.join context for distributed data parallel
        for epoch in range(start_epoch, end_epoch):
            torch.cuda.empty_cache()
            self.epoch = epoch
            self.train_dataset.set_epoch(epoch)
            for batch_id, data in enumerate(self.tr_loader):
                torch.cuda.empty_cache()
                clr = self.optim.param_groups[0]['lr']
                tr_record_dict = {}
                tr_record_dict['lr']=clr
                b = data[0].size(0)
                #if b != self.batch_size:
                #    continue
                train_data = (d.to(self.device) for d in data)
                loss, detail_loss = self.model(train_data)
                self.optim.zero_grad()
                loss.backward()
                grad_norm = clip_grad_norm_(self.model.parameters(), 5)
                if torch.isfinite(grad_norm):
                    self.optim.step()
                else:
                    self.recorder.info("!!! INFINITE grad in epoch: {}, batch_id: {}".format(
                            epoch, batch_id
                        )
                    )
                self.scheduler.step()
                self.global_step += 1
                tr_record_dict['total_loss'] = loss
                tr_record_dict.update(detail_loss)
                if self.rank == 0:
                    self.record_step({'train': tr_record_dict})

            cv_record_dict = self.cross_valid()
            if self.rank == 0:
                self.record_step({'cv': cv_record_dict})
                self.record_epoch(cv_record_dict, record_by_loss='phn_ctc_loss')
        
        if self.rank == 0:
            self.print_best_checkpoints()
                    
    def run(self, step):
        # train step
        if step <= 0:
            self.backup_configs()

        if step <= 1:
            self.init_data_loaders()
            self.init_opt_model()
            self.train()

        # avg model step
        if (step <= 2) and (self.rank == 0):
            self.avg_model()

if __name__ == '__main__':
    args = get_args()
    YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['MASTER_ADDR'] = '127.0.0.1' # only support gloo in single machine with multi-deivce
                                            # multi-machine with mulit-device should change this param
                                            # as the correct ip 
    os.environ['MASTER_PORT'] = args.port # check whether this port has been occupied.
    config = yaml.load(open(args.config),Loader=yaml.FullLoader)
    from wesep.models import get_model
    model_arch_name = config['model_arch']
    model_arch = get_model(model_arch_name)
    model = model_arch
    # torch.multiprocessing.set_sharing_strategy('shared_memory')
    trainer = Trainer(
        model, config, world_size=args.world_size, rank=args.rank,
    )
    trainer.run(args.step)
