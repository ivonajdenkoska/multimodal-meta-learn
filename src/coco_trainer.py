import logging
import os
import time

import torch
from torch import optim
from torch.nn import functional as F
from tqdm import tqdm
from transformers import get_constant_schedule_with_warmup

from meta_learner import MetaLearner
from utils import *

PROJECT_ROOT = str(Path.cwd().parent)  # project path
LOG_PATH = PROJECT_ROOT + "/logs/coco_training/"


class CocoTrainer:
    def __init__(self, args, train_loader, val_loader, device):
        super(CocoTrainer, self).__init__()
        self.experiment = str(args.coco_trained_model_id) + '_coco_training'

        self.prefix_length = args.prefix_length
        self.seq_len = args.seq_len
        self.num_epochs = args.coco_num_epochs
        self.warm_up_steps = args.warm_up_steps
        self.lr = args.lr
        self.device = device
        self.model = MetaLearner(self.prefix_length, self.seq_len, clip_model_type=args.clip_model_type)
        self.model.to(device)
        print(f"The model has {count_model_params(self.model.parameters())} trainable parameters.")

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.early_stopping = EarlyStopping(args)
        self.optimizer = optim.AdamW(params=self.model.parameters(), lr=args.lr)
        self.scheduler = get_constant_schedule_with_warmup(self.optimizer, num_warmup_steps=self.warm_up_steps)
        logging.basicConfig(filename=LOG_PATH + self.experiment + '.log', level=logging.INFO)

    def train(self):
        logging.info('Training started')
        train_losses, val_losses = [], []
        start_train = time.time()
        grad_clip = 1

        for epoch_id in range(0, self.num_epochs):
            print('------ START Epoch [{}/{}] ------'.format(epoch_id + 1, self.num_epochs))
            self.model.train()
            start = time.time()

            with tqdm(desc='Epoch [{}/{}] - training'.format(str(epoch_id + 1), self.num_epochs), unit='it',
                      total=len(self.train_loader), position=0, leave=True) as pbar:
                for batch_id, batch in enumerate(self.train_loader):
                    loss = self.forward_batch(batch)
                    self.optimizer.zero_grad()
                    loss.backward()
                    # Gradient clipping
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=grad_clip)

                    self.optimizer.step()
                    self.scheduler.step()
                    train_losses.append(loss.item())
                    pbar.set_postfix(loss=loss.item())
                    pbar.update()

                with torch.no_grad():
                    self.model.eval()
                    for batch_id, batch in enumerate(self.val_loader):
                        val_loss = self.forward_batch(batch)
                        val_losses.append(val_loss.item())

            end = time.time()
            train_log = "Training: Epoch [{}/{}], Mean Epoch Loss: {:.4f}, LR = {}" \
                .format(epoch_id + 1, self.num_epochs, np.mean(train_losses), self.optimizer.param_groups[0]['lr'])
            val_log = "Validation: Epoch [{}/{}], Mean Epoch Loss: {:.4f}, LR = {}" \
                .format(epoch_id + 1, self.num_epochs, np.mean(val_losses), self.optimizer.param_groups[0]['lr'])

            print(train_log)
            print(val_log)

            logging.info(train_log)
            logging.info(val_log)
            print('Total time train + val of epoch {}: {:.4f} seconds'.format(epoch_id + 1, (end - start)))
            print('------ END Epoch [{}/{}] ------'.format(epoch_id + 1, self.num_epochs))
            self.early_stopping(val_loss=np.mean(val_losses), model=self.model.mapper_net)
            if self.early_stopping.check_early_stop:
                print("Early stopping ...")
                logging.info("Early stopping ...")
                break

        print("End of training.")
        end_train = time.time()
        train_time_info = 'Total training took {:.4f} minutes'.format((end_train - start_train) / 60)

        print(train_time_info)
        logging.info(train_time_info)

        return self.model.mapper_net

    def forward_batch(self, batch):
        image_preprocessed = batch[0].to(self.device)
        caption = batch[2].to(self.device)
        caption_mask = batch[3].to(self.device)
        logits = self.model(image_preprocessed, caption, caption_mask, list(self.model.mapper_net.parameters()),
                            get_pred_tokens=False)
        loss = F.cross_entropy(logits.reshape(-1, len(self.model.gpt_tokenizer)), caption.flatten(),
                               ignore_index=self.model.gpt_tokenizer.pad_token_type_id)
        return loss


class EarlyStopping:
    """ Adapted from:
    Title: Early Stopping for PyTorch
    Availability: https://github.com/Bjarten/early-stopping-pytorch """
    """ Early stops the training if validation loss doesn't improve after a given patience """

    def __init__(self, args):
        # How long to wait after last time validation loss improved
        self.patience = args.early_stop_patience
        # Minimum change in the monitored quantity to qualify as an improvement
        self.delta = args.delta
        self.counter = 0
        self.best_score = None
        self.check_early_stop = False
        self.val_loss_min = np.Inf
        self.model_name = args.coco_trained_model_name

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print("Early stopping counter: {}/{}".format(self.counter, self.patience))
            logging.info("Early stopping counter: {}/{}".format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.check_early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease """
        print("Validation loss decreased ({:.4f}  --> {:.4f}). Saving model {} ..."
              .format(self.val_loss_min, val_loss, self.model_name))
        logging.info("Validation loss decreased ({:.4f}  --> {:.4f}). Saving model {} ..."
                     .format(self.val_loss_min, val_loss, self.model_name))
        torch.save(model.state_dict(), os.path.join(PROJECT_ROOT, "models", self.model_name))
        self.val_loss_min = val_loss


def count_model_params(model_parameters):
    params = list(filter(lambda p: p.requires_grad, model_parameters))
    params_summed = sum(p.numel() for p in params)
    return params_summed