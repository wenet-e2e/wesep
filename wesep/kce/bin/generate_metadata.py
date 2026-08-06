import argparse
import os
import copy
import yaml
import torch
import numpy as np
import random
import tqdm

from torch.utils.data import DataLoader
from wesep.kce.data.data_loader import build_dataset
from wesep.kce.yamlinclude import YamlIncludeConstructor
from wesep.kce.utils import read_list


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        required=True,
        help="config file in yaml format e.g. config/ref.yaml"
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

    args = parser.parse_args()
    return args


class MetadataGenerator():
    def __init__(
        self,
        config: dict,
        rank: int,
        world_size: int,
        random_seed=2022,
        #args: argparse.Namespace
    ):
        # init config info
        self.config = config

        self.save_dir = config['save_dir']
        self.data_config = config['data_config']

        self.rank = rank
        self.seed = random_seed
        self.device = torch.device('cuda')
        self.world_size = world_size
        if (not os.path.isdir(self.save_dir)) and (self.rank == 0):
            try:
                os.makedirs(self.save_dir)
            except:
                raise FileNotFoundError("can not create metadata save dir: {}".format(self.save_dir))
            
        self.set_seed(random_seed)

    def set_seed(self, random_seed: int):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        random.seed(random_seed)
        np.random.seed(random_seed)

    def backup_configs(self):
        for k, v in self.config.items():
            print("{} config: {}".format(k.upper(), v))
        
        if (self.rank == 0) and (self.data_config['start_epoch'] == 0): 
            ef = open("{}/args.yaml".format(self.save_dir), 'w')
            yaml.dump(self.config, ef)

    def init_data_loaders(self):
        num_workers = self.data_config.get('num_workers', 10)

        train_list_file = self.data_config['train_list']
        train_list = read_list(train_list_file, split_cv=False)

        self.num_samples = len(train_list)
    
        self.train_dataset = build_dataset(
            copy.deepcopy(self.data_config),
            train_list,
            tag_of_dataset='train',
        )

        self.tr_loader = DataLoader(
            self.train_dataset,
            batch_size=None,
            num_workers=num_workers
        )

        if self.data_config['start_epoch'] == 0:
            print("Num **Test** samples: {}".format(len(train_list)))
            print("Num Worker: {}".format(num_workers))

    def train(self):
        start_epoch = self.data_config['start_epoch']
        end_epoch = self.data_config['epoch']

        #TODO: add model.join context for distributed data parallel
        for epoch in range(start_epoch, end_epoch):
            torch.cuda.empty_cache()
            self.epoch = epoch
            self.train_dataset.set_epoch(epoch)
            for batch_id, data in tqdm.tqdm(enumerate(self.tr_loader)):
                if batch_id % 100 == 0:
                    print(f'{self.rank}: {batch_id}')
                    
    def run(self):
        self.backup_configs()
        self.init_data_loaders()
        self.train()


if __name__ == '__main__':
    args = get_args()
    #this line support load yaml config file recursively e.g.
    # config.yaml
    # item1: value1
    # item2: !include "config2.yaml"
    # YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader)
    YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['MASTER_ADDR'] = '127.0.0.1' # only support gloo in single machine with multi-deivce
                                            # multi-machine with mulit-device should change this param
                                            # as the correct ip 
    os.environ['MASTER_PORT'] = args.port # check whether this port has been occupied.
    config = yaml.load(open(args.config),Loader=yaml.FullLoader)

    metadataGenerator = MetadataGenerator(
        config,
        world_size=args.world_size,
        rank=args.rank,
    )
    metadataGenerator.run()
