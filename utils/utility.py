import os
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import numpy as np
import os.path as osp
import yaml
from collections import OrderedDict
from shutil import copyfile
import pickle
import warnings

try:
    import neptune
except Exception:
    print('Neptune is not installed.')


class checkpoint():
    def __init__(self, args):
        self.args = args
        self.log = torch.Tensor()
        self.since = datetime.datetime.now()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        def _make_dir(path):
            if not os.path.exists(path):
                os.makedirs(path)

        ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        if args.load == '':
            if args.save == '':
                args.save = now
            self.dir = '/content/gdrive/MyDrive/LightMBN_Save' + '/experiment/' + args.save
        else:
            self.dir = '/content/gdrive/MyDrive/LightMBN_Save' + '/experiment/' + args.save
            if not os.path.exists(self.dir):
                args.load = ''
            args.save = args.load

        self.local_dir = None
        if ROOT_PATH[:11] == '/content/dr':
            self.dir = osp.join('/content/drive/Shareddrives/Colab',
                                self.dir[self.dir.find('experiment'):])
            self.local_dir = ROOT_PATH + '/experiment/' + self.dir.split('/')[-1]
            _make_dir(self.local_dir)

        _make_dir(self.dir)

        last_folder = os.path.basename(args.datadir.rstrip('/\\'))
        if '_' in last_folder:
            self.fold = last_folder.split('_')[-1].upper()
        else:
            self.fold = 'A'
        self.log_filename = f"{args.model}_{args.data_train}_{self.fold}_log.txt"
        self.map_log_filename = f"{args.model}_{args.data_train}_{self.fold}_map_log.pt"
        self.config_filename = f"{args.model}_{args.data_train}_{self.fold}_config.yaml"
        self.model_latest_filename = f"{args.model}_{args.data_train}_{self.fold}_model-latest.pth"
        self.model_best_filename = f"{args.model}_{args.data_train}_{self.fold}_model-best.pth"

        map_log_path = os.path.join(self.dir, self.map_log_filename)
        if os.path.exists(map_log_path):
            self.log = torch.load(map_log_path)

        print(f'Experiment results will be saved in {self.dir}')

        log_path = os.path.join(self.dir, self.log_filename)
        open_type = 'a' if os.path.exists(log_path) else 'w'
        self.log_file = open(log_path, open_type)

        try:
            exp = neptune.init(args.nep_name, args.nep_token)
            if args.load == '':
                self.exp = exp.create_experiment(name=self.dir.split('/')[-1],
                                                 params=vars(args))
                args.nep_id = self.exp.id
            else:
                self.exp = exp.get_experiments(id=args.nep_id)[0]
            print(self.exp.id)
        except Exception:
            pass

        config_path = os.path.join(self.dir, self.config_filename)
        with open(config_path, open_type) as fp:
            dic = vars(args).copy()
            for k in ['load', 'save', 'pre_train', 'test_only', 're_rank', 'activation_map', 'nep_token']:
                dic.pop(k, None)
            yaml.dump(dic, fp, default_flow_style=False)

        if self.local_dir is not None:
            copyfile(config_path, os.path.join(self.local_dir, self.config_filename))

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False, end='\n'):
        time_elapsed = (datetime.datetime.now() - self.since).seconds
        log = log + f' Time used: {time_elapsed // 60} m {time_elapsed % 60} s'
        print(log, end=end)
        if end != '':
            self.log_file.write(log + end)
            try:
                t = log.find('Total')
                m = log.find('mAP')
                r = log.find('rank1')
                self.exp.log_metric('batch loss', float(log[t + 7:t + 12])) if t > -1 else None
                self.exp.log_metric('mAP', float(log[m + 5:m + 11])) if m > -1 else None
                self.exp.log_metric('rank1', float(log[r + 7:r + 13])) if r > -1 else None
            except Exception:
                pass

        if refresh:
            self.log_file.close()
            self.log_file = open(os.path.join(self.dir, self.log_filename), 'a')
            if self.local_dir is not None:
                copyfile(os.path.join(self.dir, self.log_filename), os.path.join(self.local_dir, self.log_filename))

    def done(self):
        self.log_file.close()

    def plot_map_rank(self, epoch):
        axis = np.linspace(1, epoch, self.log.size(0))
        labels = ['mAP', 'rank1', 'rank3', 'rank5', 'rank10']
        fig = plt.figure()

        title = f'{self.args.model} on {self.args.data_test} ({self.fold}-fold)'
        plt.title(title)

        for i in range(len(labels)):
            plt.plot(axis, self.log[:, i + 1].numpy(), label=labels[i])

        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('mAP/rank')
        plt.grid(True)

        pdf_name = f'{self.args.model}_{self.args.data_test}_{self.fold}_result.pdf'
        plt.savefig(os.path.join(self.dir, pdf_name), dpi=600)
        plt.close(fig)

    def save_checkpoint(self, state, save_dir, is_best=False, remove_module_from_keys=False):
        def mkdir_if_missing(dirname):
            if not osp.exists(dirname):
                os.makedirs(dirname)

        mkdir_if_missing(save_dir)
        if remove_module_from_keys:
            state_dict = state['state_dict']
            new_state_dict = OrderedDict((k[7:] if k.startswith('module.') else k, v)
                                         for k, v in state_dict.items())
            state['state_dict'] = new_state_dict

        torch.save(state, os.path.join(save_dir, self.model_latest_filename))
        self.write_log(f'[INFO] Checkpoint saved to "{os.path.join(save_dir, self.model_latest_filename)}"')

        if is_best:
            torch.save(state['state_dict'], os.path.join(save_dir, self.model_best_filename))

        if 'log' in state:
            torch.save(state['log'], os.path.join(save_dir, self.map_log_filename))

    def load_checkpoint(self, fpath):
        if fpath is None:
            raise ValueError('File path is None')
        if not osp.exists(fpath):
            raise FileNotFoundError(f'File is not found at "{fpath}"')
        map_location = None if torch.cuda.is_available() else 'cpu'
        try:
            checkpoint = torch.load(fpath, map_location=map_location)
        except UnicodeDecodeError:
            pickle.load = partial(pickle.load, encoding="latin1")
            pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
            checkpoint = torch.load(fpath, pickle_module=pickle, map_location=map_location)
        except Exception:
            print(f'Unable to load checkpoint from "{fpath}"')
            raise
        return checkpoint

    def load_pretrained_weights(self, model, weight_path):
        checkpoint = self.load_checkpoint(weight_path)
        state_dict = checkpoint.get('state_dict', checkpoint)
        model_dict = model.state_dict()
        new_state_dict = OrderedDict()
        matched_layers, discarded_layers = [], []

        for k, v in state_dict.items():
            k = k.replace('module.', '')
            k = k.replace('model.', '')
            if k in model_dict and model_dict[k].size() == v.size():
                new_state_dict[k] = v
                matched_layers.append(k)
            else:
                discarded_layers.append(k)

        model_dict.update(new_state_dict)
        model.load_state_dict(model_dict)

        if not matched_layers:
            warnings.warn(f'No matched layers found in "{weight_path}"')
        else:
            self.write_log(f'[INFO] Successfully loaded pretrained weights from "{weight_path}"')
            if discarded_layers:
                print(f'Discarded layers: {discarded_layers}')

    def resume_from_checkpoint(self, fpath, model, optimizer=None, scheduler=None):
        self.write_log(f'[INFO] Loading checkpoint from "{fpath}"')
        checkpoint = self.load_checkpoint(fpath)
        self.load_pretrained_weights(model, fpath)
        self.write_log('[INFO] Model weights loaded')

        if optimizer and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            self.write_log('[INFO] Optimizer loaded')

        if scheduler and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
            self.write_log('[INFO] Scheduler loaded')

        start_epoch = checkpoint['epoch']
        self.write_log(f'[INFO] Last epoch = {start_epoch}')

        if 'rank1' in checkpoint:
            self.write_log(f'[INFO] Last rank1 = {checkpoint["rank1"]:.1%}')

        if 'log' in checkpoint:
            self.log = checkpoint['log']

        return start_epoch, model, optimizer
