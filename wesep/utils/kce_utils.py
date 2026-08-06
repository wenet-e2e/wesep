import logging
import torch
import re
import sklearn.metrics as skmt
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import _LRScheduler


log_level = {
    'INFO': logging.INFO,
    'DEBUG': logging.DEBUG,
    'WARN': logging.WARNING,
    'ERROR': logging.ERROR
}

def isdigit(string):
    if not isinstance(string, str):
        result = False
    else:
        pattern = re.compile(r'^\d+(\.\d+)?$')
        result = bool(pattern.match(string))
    return result

def beam_search(hyp, beam_size):
    pass

def compute_eer_skleanr(y_true, y_score):

    fpr, tpr, threshold = skmt.roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr

    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    eer = (eer_1 + eer_2) / 2
    return eer, (eer_threshold, fpr, fnr)

def compute_topk(scores, k=1):
    if not isinstance(scores, torch.Tensor):
        scores = torch.tensor(scores)
    
    sorted_score, top_idx = torch.sort(scores, descending=True, dim=-1)
    if scores.dim() > 1:
        top_idx = top_idx[:,:k]
    else:
        top_idx = top_idx[:k]
    return sorted_score, top_idx


def compute_eer(scores, labels):
    if isinstance(scores, list) is False:
        scores = list(scores)
    if isinstance(labels, list) is False:
        labels = list(labels)

    target_scores = []
    nontarget_scores = []

    for item in zip(scores, labels):
        if item[1] == 1:
            target_scores.append(item[0])
        else:
            nontarget_scores.append(item[0])

    target_size = len(target_scores)
    nontarget_size = len(nontarget_scores)
    target_scores = sorted(target_scores)
    nontarget_scores = sorted(nontarget_scores)

    target_position = 0
    for i in range(target_size-1):
        target_position = i
        nontarget_n = nontarget_size * float(target_position) / target_size
        nontarget_position = int(nontarget_size - 1 - nontarget_n)
        if nontarget_position < 0:
            nontarget_position = 0
        if nontarget_scores[nontarget_position] < target_scores[target_position]:
            break
    th = target_scores[target_position]
    eer = target_position * 1.0 / target_size
    return eer, th


def plot_weight(writer):
    # use to visualize model weigh it's a temporary function 
    ckpt = torch.load('output.feats.pt', map_location='cpu')
    netout = ckpt['output']
    pkey = netout.keys()
    netweight = ckpt['weight']
    
    pdict = make_lexicon()
    for pid in pdict.keys():
        if not pid in pkey:
            continue
        nout = netout[pid]
        w = netweight[pid]
        w = w.reshape(-1)
        wmin = np.min(w)
        frames = nout.size(0)
        if frames < 5000:
            continue
        pmean = nout.mean(dim=0).to('cpu').numpy()
        pmin = np.min(pmean)
        if pmin < wmin:
            m = pmin
        else:
            m = wmin
        pmean -= m
        w -= m
        phone = pdict[pid]
        for i in range(256):
            writer.add_scalars(
                main_tag="{}".format(phone),
                tag_scalar_dict={
                    'net_emb_mean'.format(phone): pmean[i],
                    'net_weight'.format(phone): w[i],
                },
                global_step=i
            )

def trim_wav(
    wavf, 
    head, 
    tail,
    sr=16000,
    save_path=None
):
    # use to trim wav to check the speech content
    wav,_ = sf.read(wavf)
    sample_per_frame = sr / 100
    head_anchor = int(head * sample_per_frame)
    tail_anchor = int(tail * sample_per_frame)
    trim_wav = wav[head_anchor: tail_anchor]
    if save_path != None:
        sf.write(save_path, trim_wav, samplerate=sr) 
        return 1
    else:
        return trim_wav

def vinterplate(matrix, deep=4):
    # visualize fbank feats interpalte it to a larger size
    # TODO: find some way to visualize fbank
    assert (
        isinstance(matrix, torch.Tensor) or \
            isinstance(matrix, np.ndarray)
    )
    if isinstance(matrix, torch.Tensor):
        d,t = matrix.size()
        new_matrix = torch.rand(d*deep, t)
    else:
        d,t = matrix.shape
        new_matrix = np.random.rand(d*deep, t)

    #new_matrix = torch.rand(d*deep, t*deep)
    for x in range(d*deep):
        #for y in range(t*deep):
            #target = matrix[x//deep][y//deep]
            target = matrix[x//deep]
            new_matrix[x] = target
    return new_matrix


# find best score path from the posteriori matrix
def compute_one_best(posteriori):
    if not isinstance(posteriori, torch.Tensor):
        assert(
            isinstance(posteriori, T)
            for T in [list, torch.Tensor, np.ndarray]
        )
        posteriori = torch.tensor(posteriori)
        device = 'cpu'
    else:
        device = posteriori.device
    _, d = posteriori.size()
    # best score for evry column and convert it to one hot format
    one_best = torch.argmax(posteriori, dim=-1)
    one_best_sidx = F.one_hot(
        one_best, 
        num_classes=d, 
        device=device
    )
    # [0.5, 0.3, 0.2]   [1, 0, 0]
    # [0.3, 0.5, 0.2] X [0, 1, 0]
    # [0.1, 0.4, 0.5]   [0, 0, 1]
    one_best_mtx = posteriori * one_best_sidx
    # [f, t, t]
    # [t, f, t]
    # [t, t, f] this is a idx matrix
    best_idx = one_best_mtx == 0
    # select element according to idx maxtrix
    one_best_path = one_best_mtx[~best_idx]

    return one_best_path


def remove_blank(
    posteriori,
    blank_id
):
    # remove a black id from rnn-t or ctc output
    if not isinstance(posteriori, torch.Tensor):
        assert(
            isinstance(posteriori, T)
            for T in [list, torch.Tensor, np.ndarray]
        )
        posteriori = torch.tensor(posteriori)
    _, max_idx = torch.max(posteriori, dim=-1)
    no_blank_idx = ~(max_idx == blank_id)

    return posteriori[no_blank_idx]
    

# splice feats i.e. add contenx for frames
def splice_feats(feats, left_context=4, right_context=4):
    if not isinstance(feats, torch.Tensor):
        feats = torch.tensor(feats)
    frames, nmel = feats.size()
    l_padding = torch.ones_like(
        torch.rand(left_context, nmel)
    )
    r_padding = torch.ones_like(
        torch.rand(right_context, nmel)
    )
    l_padding *= feats[0]
    r_padding *= feats[-1]
    feats = torch.cat(
        [l_padding, feats, r_padding], dim=0
    )
    splice_v = []
    for i in range(left_context+right_context+1):
        v = feats[i:frames+i]
        splice_v.append(v)
    feats = torch.cat([v for v in splice_v], dim=-1)
    return feats

# read data list; split train and cv set
def read_list(list_file, split_cv=False, shuffle=True):
    d_list = []
    import random
    with open(list_file, encoding='utf-8') as lf:
        for line in lf.readlines():
            d_list.append(line.strip()) 
    if shuffle:
        random.shuffle(d_list)

    if split_cv:
        n = len(d_list)
        if n > 500000:
            n_cv = 5000
        else:
            n_cv = int(n*0.1)
        cv = random.choices(d_list, k=n_cv)
        tr = [x for x in d_list if x not in cv]
        return tr, cv
    else:
        return d_list

# make lexicon
def make_lexicon(phones='data/txt/phones.txt'):
    pdict = {}
    with open(phones) as pf:
        for line in pf.readlines():
            p, pid = line.strip().split(" ")
            pdict[int(pid)-1] = p
    return pdict

# make dict
def make_dict_from_file(files):
    rdict = {}
    with open(files) as f:
        for line in f.readlines():
            k,v = line.strip().split(" ", maxsplit=1)
            rdict[k] = v
    return rdict

# remove duplicates and blank for one sequence(CTC decode)
def remove_duplicates_and_blank(hyp, blank_id=0):
    if isinstance(hyp, torch.Tensor):
        hyp = [x.item() for x in hyp]
    new_hyp = []
    cur = 0
    while cur < len(hyp):
        if hyp[cur] != blank_id:
            new_hyp.append(hyp[cur])
        prev = cur
        while cur < len(hyp) and hyp[cur] == hyp[prev]:
            cur += 1
    return new_hyp

# warm up scheduler
class WarmUpLR(_LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_steps = 20, # org 2000
        last_epoch=-1
    ):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def __repr__(self):
        return f"{self.__class__.__name}(warmup_steps={self.warmup_steps})"

    def get_lr(self):
        step_num = self.last_epoch + 1

        return {
            lr * self.warmup_steps ** 0.5 * min(
                step_num ** -0.5, step_num * self.warmup_steps ** -1.5
            )
            for lr in self.base_lrs
        }

    def set_step(self, step):
        self.last_epoch = step

# recorder 
class Recorder():
    def __init__(self, log_config):

        self.log = logging
        base_config = log_config['log_config']
        l_level = base_config['level']
        base_config['level'] = log_level[l_level]
        self.log.basicConfig(**base_config)
        self.exp_dir = log_config['exp_dir']
        self.checkpoint_name = log_config['checkpoint_name']
        self.log_interval = log_config['log_interval']
        self.loss_recorder = {
            "train": {},
            "cv": {}
        }

    def info(self, var):
        if not isinstance(var, str):
            self.log.info("record variabel: {}".format(var))
        self.log.info(var)

    def record_best(
        self, loss_type, loss,
        model, opt, epoch, step, tag='train'
    ):
        if loss < self.loss_recorder[tag][loss_type]['best_loss']:
            self.loss_recorder[tag][loss_type]['best_model'] = {
                'model': model,
                'loss': loss,
                'opt': opt,
                'epoch': epoch,
                'step': step
            }
            self.loss_recorder[tag][loss_type]['best_loss'] = loss
        else:
            pass
    
    def record_detail(
        self, loss_detail, epoch,
        step, model, opt, tag='train'
    ):
        if not isinstance(loss_detail, dict):
            loss = self.detach_torch(value)
            self.info(">>>{} Epoch: {}, step: {} loss is: {}".format(
                    tag.upper(), epoch, step, loss
                )
            )
            #self.record_best("total_loss", loss, model, opt, epoch, step, tag)
        else:
            record_line = ">>> {:<5} Epoch:{:<3} step:{:<8}".format(
                tag, epoch, step
            )
            for key, value in loss_detail.items():
                value = self.detach_torch(value)
                if key == "lr":
                    one = "{}: {:5f}".format(key, value)
                    record_line += one
                    continue

                if key not in self.loss_recorder[tag].keys():
                    self.loss_recorder[tag][key] = {
                        'loss': [value],
                        'best_loss': float('inf'),
                        'best_model': None
                    }
                else:
                    self.loss_recorder[tag][key]['loss'].append(value)
                #self.record_best(key, value, model, opt, epoch, step, tag)
                one = " {}: {:.5f} ".format(
                    key,
                    value
                )
                record_line += one
            if (step % self.log_interval == 0) or (tag == 'cv'):
                self.log.info(record_line)

    def clean_epoch(self):
        #for tag in ['cv', 'train']:
        for tag in ['train']:
            sub_key = list(self.loss_recorder[tag].keys())
            for sk in sub_key:
                self.loss_recorder[tag][sk]['best_loss'] = float('inf')
                self.loss_recorder[tag][sk]['best_model'] = None
        
    def record_epoch(
        self, epoch, step,
        model, opt, cv_loss
    ):
        for tag in ['train', 'cv']:
            for key, value in self.loss_recorder[tag].items():
                loss_curve = value['loss']
                plot_x = [x for x in range(len(loss_curve))]
                plt.figure()
                plt.plot(plot_x, loss_curve)
                plt.title("loss: {}".format(key))
                plt.savefig(
                    "{}/{}_{}.png".format(self.exp_dir, tag, key), 
                    dpi=600
                )
                plt.close()
        self.save_state(model, opt, epoch, step, cv_loss)
        self.clean_epoch()
    
    def save_state(self, model, opt=None, epoch=None, step=None, cv_loss=None):
        torch.save(
            {
                'model': model,
                'epoch': epoch,
                'step': step,
                'opt': opt,
                'cv_loss': cv_loss
            },
            "{exp_dir}/{checkpoint_name}_{epoch}.pt".format(
                exp_dir=self.exp_dir,
                checkpoint_name=self.checkpoint_name,
                epoch=epoch
            )
        )

    def record_tmp(self, loss):
        for key, value in loss.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu()
            self.log.info("loss_{}: {}".format(key, value))

    def detach_torch(self, value):
        if not isinstance(value, torch.Tensor):
            return value
        if value.device.type != 'cpu':
            return value.detach().cpu().item()

    def record_test_result(self, r):
        pass
    
    def plot_roc(self, r):
        pass

    def visual(self):
        #TODO: matplotlib plot loss curve 
        #TODO: temporarily implement in record epoch
        pass


# ---------------------------------------------------------------------------
# Evaluation utilities (moved from wesep.kce.dev.utils)
# ---------------------------------------------------------------------------
import random
import os as _os
import json as _json
from logging.handlers import RotatingFileHandler
from typing import List as _List, Optional as _Optional


def setup_logger(log_file_dir: str, rank: int) -> None:
    _os.makedirs(log_file_dir, exist_ok=True)
    file_handler = RotatingFileHandler(
        _os.path.join(log_file_dir, f'infer.{rank}.log'),
        maxBytes=50*1024*1024, backupCount=5, encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(
        logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s')
    )
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console)


def ctc_map(sequence: _List[int], blank_id: int = 0) -> _List[int]:
    result = []
    prev = None
    for token in sequence:
        if token == blank_id or token == prev:
            if token == blank_id:
                prev = None
            continue
        result.append(token)
        prev = token
    return result


def is_sublist(main_list: _List, sublist: _List) -> bool:
    n, m = len(main_list), len(sublist)
    if m > n:
        return False
    for i in range(n - m + 1):
        if main_list[i: i + m] == sublist:
            return True
    return False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def batch_data_to_device(data_list: _List[torch.Tensor], device: torch.device) -> _List[torch.Tensor]:
    return [d.to(device) for d in data_list]


def find_keyword_sublist_span(keyword: _List[int], word_labels: _List[_List[int]]) -> _Optional[tuple]:
    flat_list = []
    index_map = []
    for i, sublist in enumerate(word_labels):
        flat_list.extend(sublist)
        index_map.extend([i] * len(sublist))
    for i in range(len(flat_list) - len(keyword) + 1):
        if flat_list[i:i + len(keyword)] == keyword:
            return index_map[i], index_map[i + len(keyword) - 1]
    return None


def log_json_info(data: dict, prefix: str = "") -> None:
    line = _json.dumps(data)
    logging.info(f"{prefix} {line}")
