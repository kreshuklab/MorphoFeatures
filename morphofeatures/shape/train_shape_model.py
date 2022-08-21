import os
import sys
import torch
import yaml
import wandb

from collections import namedtuple
from tqdm import tqdm
from pytorch_metric_learning import losses

from morphofeatures.shape.network import DeepGCN
from morphofeatures.shape.data_loading.loader import get_train_val_loaders


class ShapeTrainer:

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device(config['device'])
        self.ckpt_dir = os.path.join(config['experiment_dir'], 'checkpoints')
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.build_loaders()
        self.reset()

    def reset(self):
        self.build_model()
        self.best_val_loss = None
        self.epoch = 0
        self.step = 0

    def build_loaders(self):
        dataset_config = self.config['data']
        loader_config = self.config['loader']
        loaders = get_train_val_loaders(dataset_config, loader_config)
        self.train_loader = loaders['train']
        self.val_loader = loaders['val']

    def build_model(self):
        self.model = DeepGCN(**config['model']['kwargs'])
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(
                self.model,
                device_ids=[i for i in range(torch.cuda.device_count())]
            )
            self.model.cuda()
        else:
            self.model = self.model.to(self.device)

        self.optimizer = getattr(
            torch.optim, self.config['optimizer']['name']
        )(self.model.parameters(), **self.config['optimizer']['kwargs'])
        self.criterion = getattr(
            losses, self.config['criterion']['name']
        )(**self.config['criterion']['kwargs'])

    def checkpoint(self, force=True):
        save = force or (self.epoch % self.config['training']['checkpoint_every'] == 0)
        if save:
            info = {
                'epoch': self.epoch,
                'iteration': self.step,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                # 'scheduler': self.scheduler.state_dict(),
                'config/model/name': self.config['model']['name'],
                'config/model/kwargs': self.config['model']['kwargs'],
            }
            ckpt_name = f'best_ckpt_iter_{self.step}.pt' if force else f'ckpt_iter_{self.step}.pt'
            ckpt_path = os.path.join(self.ckpt_dir, ckpt_name)
            torch.save(info, ckpt_path)

    def train_epoch(self):
        self.model.train()
        # generator = iter(self.train_loader)
        # for iteration in range(len(self.train_loader)):
        generator = iter(self.val_loader)
        for iteration in tqdm(range(len(self.val_loader)), desc='Iteration'):
            try:
                data = next(generator)
            except (IndexError, TypeError) as e:
                print(e)
                continue
            out, h = self.model(data['points'], data['features'])
            labels = torch.arange(out.size(0) // 2) \
                          .repeat_interleave(2) \
                          .to(self.device)

            loss = self.criterion(out, labels)
            if torch.isnan(loss).item():
                print(f'Loss: {loss.item()}')
                continue

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            wandb.log({'training/loss': loss.item()}, step=self.step)
            self.step += 1

    def validate_epoch(self):
        if self.epoch % self.config['training']['validate_every'] != 0:
            return
        self.model.eval()
        total_loss = 0.0
        for data in tqdm(self.val_loader):
            out, h = self.model(data['points'], data['features'])
            labels = torch.arange(out.size(0) // 2) \
                          .repeat_interleave(2) \
                          .to(self.device)
            total_loss += self.criterion(out, labels).item()
        avg_loss = total_loss / len(self.val_loader)

        if self.best_val_loss is None or avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.checkpoint(True)

        wandb.log({'validation': {'average_loss': avg_loss}})

    def train(self):
        for epoch_num in tqdm(range(self.config['training']['epochs']), desc='Epochs'):
            self.train_epoch()
            self.validate_epoch()
            self.scheduler.step(epoch_num)
            self.checkpoint(False)
            self.epoch += 1

    def run(self):
        with wandb.init(project='MorphoFeatures'):
            self.validate_epoch()
            self.train()


if __name__ == '__main__':
    path_to_config = sys.argv[1]
    with open(path_to_config, 'r') as f: 
        config = yaml.load(f, Loader=yaml.FullLoader)

    trainer = ShapeTrainer(config)
    trainer.run()
