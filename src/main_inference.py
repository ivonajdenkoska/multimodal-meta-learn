import argparse
import datetime

from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

import utils
from data_loaders import *
from meta_trainer import MetaTrainer

PATH = str(Path.cwd().parent.parent.parent)  # root directory
LOG_PATH = PROJECT_ROOT + "/logs/"
MODELS_PATH = PROJECT_ROOT + "/models/"

torch.manual_seed(222)
torch.cuda.manual_seed_all(222)
np.random.seed(222)
device = set_device()


def main_inference():
    gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # gpt_tokenizer.add_tokens(way_2 if args.n_way == 2 else way_5)

    experiment_id = "{}_{}_{}-way_{}-shot".format(args.experiment_id, args.data_root, args.n_way, args.k_spt)
    meta_trainer = MetaTrainer(args, experiment_id, is_pretrained=True, new_words=False)

    params = list(filter(lambda p: p.requires_grad, meta_trainer.model.parameters()))
    params_summed = sum(p.numel() for p in params)
    print("Total num of params: {} ".format(params_summed))
    log_file_path = LOG_PATH + "log_{}.txt".format(experiment_id)
    utils.write_data_to_txt(log_file_path, "Experiment ID: {} Date: {}, {}-way, {}-shot (support), {}-shot (query)\n"
                            .format(experiment_id, datetime.datetime.now(), args.n_way, args.k_spt, args.k_qry))

    meta_test_loader = MetaTestTaskDataLoader(root=args.data_root, mode='test', n_way=args.n_way, k_shot=args.k_spt,
                                              k_query=1, batchsz=100, resize=args.img_size, seq_len=args.seq_len,
                                              repeats=args.repeats, tokenizer=gpt_tokenizer,
                                              clip_model_type=args.clip_model_type, prefix_length=args.prefix_length)
    db_test = DataLoader(meta_test_loader, batch_size=1, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    print("Num. of tasks: {}".format(len(db_test)))
    accs_all_test = []

    for step, (x_spt, y_spt, y_spt_mask, x_qry, y_qry, y_qry_mask, y_qry_answer, qry_img_id) in enumerate(db_test):
        x_spt, y_spt, y_spt_mask, x_qry, y_qry, y_qry_mask, y_qry_answer, qry_img_id = x_spt.squeeze(0).to(device), \
                                                                                       y_spt.squeeze(0).to(device), \
                                                                                       y_spt_mask.squeeze(0).to(device), \
                                                                                       x_qry.to(device), \
                                                                                       y_qry.to(device), \
                                                                                       y_qry_mask.to(device), \
                                                                                       y_qry_answer.to(device), \
                                                                                       qry_img_id

        accs = meta_trainer.finetunning(x_spt, y_spt, y_spt_mask, x_qry, y_qry, y_qry_mask, y_qry_answer, qry_img_id)
        accs_all_test.append(accs)
        print("------ Meta-test {}-way, {}-shot ({}-query) ------".format(args.n_way, args.k_spt, args.k_qry))
        print("Step: {} \tTest acc: {}\n".format(step, accs))
        utils.write_data_to_txt(file_path=log_file_path, data="Step: {} \tTest acc: {}\n".format(step, accs))

    accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
    print("FINAL: \tTest acc: {} \n".format(accs))
    utils.write_data_to_txt(file_path=log_file_path, data="FINAL: \tTest acc: {} \n".format(accs))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--experiment_id', type=int, default=5)
    argparser.add_argument('--data_root', type=str, default='real_name_mi')
    argparser.add_argument('--project_root', type=str, default=PATH)
    argparser.add_argument('--clip_model_type', type=str, default="ViT-B/32")
    argparser.add_argument('--epoch', type=int, help='epoch number', default=60000)
    argparser.add_argument('--n_way', type=int, help='n way', default=2)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=1)
    argparser.add_argument('--repeats', type=int, help='repeats for support set', default=0)
    argparser.add_argument('--img_size', type=int, help='img_size', default=224)
    argparser.add_argument('--seq_len', type=int, default=14)  # for padding batch
    argparser.add_argument('--prefix_length', type=int, default=8)
    argparser.add_argument('--mapper_type', type=str, default="ATT")  # MLP or ATT

    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=5)
    argparser.add_argument('--num_workers', type=int, default=8)

    args = argparser.parse_args()

    main_inference()
