import os
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from meta_learner import MetaLearner
from utils import *

PATH = str(Path.cwd().parent)
MODELS_PATH = PATH + "/models/"


class MetaTrainer(nn.Module):
    """
    Adapted from https://github.com/dragen1860/MAML-Pytorch/blob/98a00d41724c133bd29619a2fb2cc46dd128a368/meta.py
    """

    def __init__(self, args, experiment_id, is_pretrained, new_words=False):
        super(MetaTrainer, self).__init__()
        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.prefix_length = args.prefix_length
        self.seq_len = args.seq_len
        self.device = set_device()
        self.model_name = "{}.pt".format(experiment_id)
        self.log_file_path = PATH + "/logs/log_{}.txt".format(experiment_id)

        self.model = MetaLearner(prefix_length=self.prefix_length, seq_len=self.seq_len,
                                 clip_model_type=args.clip_model_type, new_words=new_words)

        # Loading pre-trained model
        if is_pretrained:
            model_dict = torch.load(MODELS_PATH + "coco_trained_model.pt", map_location=torch.device(self.device))
            self.model.mapper_net.load_state_dict(model_dict)
        self.model.to(self.device)
        self.meta_optim = optim.AdamW(self.model.mapper_net.parameters(), lr=self.meta_lr)
        self.pad_token_id = self.model.gpt_tokenizer.eos_token_id

    def forward(self, x_spt, y_spt, y_spt_mask, id_spt, x_qry, y_qry, y_qry_mask, id_qry):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        task_num = x_spt.shape[0]
        querysz = x_qry.shape[1]

        # losses_q[i] is the loss on step i
        losses_q = torch.zeros((self.update_step + 1)).to(self.device)
        corrects = [0 for _ in range(self.update_step + 1)]

        for i in range(task_num):
            # 1. run the i-th task and compute loss for k=0
            logits = self.model(x_spt[i], y_spt[i], y_spt_mask[i], list(self.model.mapper_net.parameters()),
                                get_pred_tokens=False)
            loss = F.cross_entropy(logits.reshape(-1, len(self.model.gpt_tokenizer)), y_spt[i].flatten(),
                                   ignore_index=self.pad_token_id)
            grad = torch.autograd.grad(loss, self.model.mapper_net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0],
                                    zip(grad, self.model.mapper_net.parameters())))
            question = y_qry[i]
            answer = y_qry[i]

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q, pred_tokens = self.model(x_qry[i], question, y_qry_mask[i],
                                                   list(self.model.mapper_net.parameters()))
                loss_q = F.cross_entropy(logits_q.reshape(-1, len(self.model.gpt_tokenizer)), answer.flatten(),
                                         ignore_index=self.pad_token_id)
                losses_q[0] += loss_q

                correct = torch.eq(pred_tokens, answer).sum().item()
                corrects[0] = corrects[0] + correct

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q, pred_tokens = self.model(x_qry[i], question, y_qry_mask[i], fast_weights)
                loss_q = F.cross_entropy(logits_q.reshape(-1, len(self.model.gpt_tokenizer)), answer.flatten(),
                                         ignore_index=self.pad_token_id)
                losses_q[1] += loss_q

                correct = torch.eq(pred_tokens, answer).sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.model(x_spt[i], y_spt[i], y_spt_mask[i], fast_weights, get_pred_tokens=False)
                loss = F.cross_entropy(logits.reshape(-1, len(self.model.gpt_tokenizer)), y_spt[i].flatten(),
                                       ignore_index=self.pad_token_id)
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(outputs=loss, inputs=fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad ()SHD
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                logits_q, pred_tokens = self.model(x_qry[i], question, y_qry_mask[i], fast_weights)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = F.cross_entropy(logits_q.reshape(-1, len(self.model.gpt_tokenizer)), answer.flatten(),
                                         ignore_index=self.pad_token_id)
                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    correct = torch.eq(pred_tokens, answer).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct

            # For ablation study
            # for name, param_value in self.model.mapper_net.named_parameters():
            #     if len(param_value.shape) == 1:
            #         nn.init.zeros_(param_value)
            #     else:
            #         nn.init.xavier_uniform_(param_value)

        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = torch.mean(losses_q[2:]) / task_num

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward(inputs=list(self.model.mapper_net.parameters()))

        # print("Grad norm before clipping: {}".format(get_grad_norms(self.model.mapper_net.parameters())))
        nn.utils.clip_grad_norm_(self.model.mapper_net.parameters(), max_norm=1)
        # print("Grad norm after clipping: {}".format(get_grad_norms(self.model.mapper_net.parameters())))

        self.meta_optim.step()
        accs = np.array(corrects) / (querysz * task_num * self.seq_len)
        losses_q_ = [round(loss.item(), 4) for loss in losses_q]

        return accs, losses_q_

    def finetunning(self, x_spt, y_spt, y_spt_mask, x_qry, y_qry, y_qry_mask, qry_answer, qry_img_id):
        querysz = len(x_qry)

        corrects = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.model
        model = deepcopy(self.model)

        # 1. run the i-th task and compute loss for k=0
        logits = model(x_spt, y_spt, y_spt_mask, fast_weights=list(model.mapper_net.parameters()), get_pred_tokens=False)
        loss = F.cross_entropy(logits.reshape(-1, len(model.gpt_tokenizer)), y_spt.flatten(),
                               ignore_index=self.pad_token_id)
        grad = torch.autograd.grad(outputs=loss, inputs=model.mapper_net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, model.mapper_net.parameters())))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q, pred_tokens = model(x_qry, y_qry, y_qry_mask, fast_weights=list(model.mapper_net.parameters()))
            # [setsz]
            correct = torch.eq(pred_tokens, qry_answer).sum().item()
            corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q, pred_tokens = model(x_qry, y_qry, y_qry_mask, fast_weights=fast_weights)

            correct = torch.eq(pred_tokens, qry_answer).sum().item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = model(x_spt, y_spt, y_spt_mask, fast_weights=fast_weights, get_pred_tokens=False)
            loss = F.cross_entropy(logits.reshape(-1, len(model.gpt_tokenizer)), y_spt.flatten(),
                                   ignore_index=self.pad_token_id)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q, pred_tokens = model(x_qry, y_qry, y_qry_mask, fast_weights=fast_weights, get_pred_tokens=True)

            with torch.no_grad():
                # pred_tokens = self.model.generate_text(logits_q)
                correct = torch.eq(pred_tokens, qry_answer).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct

        for idx in range(x_qry.shape[0]):
            gt_answer = self.model.gpt_tokenizer.decode(qry_answer[idx], skip_special_tokens=True).strip()
            pred_answer = self.model.gpt_tokenizer.decode(pred_tokens[idx], skip_special_tokens=True).strip()
            write_data_to_txt(self.log_file_path, ("Img: {}, GT answer: {}, Pred. answer: {}\n"
                                                   .format(qry_img_id, gt_answer, pred_answer)))

        del model
        accs = np.array(corrects) / (querysz * self.seq_len)

        return accs

    def save_mapper_model(self):
        torch.save({'mapper_net': self.model.mapper_net.state_dict()}, os.path.join(MODELS_PATH, "{}"
                                                                                    .format(self.model_name)))
        print("Model saved on path {}".format(MODELS_PATH))

    def load_model(self):
        model_dict = torch.load(MODELS_PATH + self.model_name, map_location=torch.device(self.device))
        self.model.mapper_net.load_state_dict(model_dict['mapper_net'])
        print("Model loaded from {}".format(MODELS_PATH))


def write_data_to_txt(file_path, data):
    if path.exists(file_path):
        with open(file_path, 'a', newline='') as file:
            file.write(data)
    else:  # Create the file
        with open(file_path, 'w') as file:
            file.write(data)
