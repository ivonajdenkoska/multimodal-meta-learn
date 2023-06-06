import argparse
import datetime

from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

from coco_trainer import CocoTrainer
from data_loaders import *
from meta_trainer import MetaTrainer
from utils import *

PATH = str(Path.cwd().parent.parent.parent)  # root of all projects
LOG_PATH = PROJECT_ROOT + "/logs/"
torch.manual_seed(222)
torch.cuda.manual_seed_all(222)
np.random.seed(222)



device = set_device()
if torch.cuda.is_available():
    print('Training on GPU!')
else:
    print('Training on CPU!')


def write_data_to_txt(file_path, data):
    if path.exists(file_path):
        with open(file_path, 'a', newline='') as file:
            file.write(data)
    else:  # Create the file
        with open(file_path, 'w') as file:
            file.write(data)


def main_non_episodic():
    device = set_device()
    gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    coco_data_train = CocoDataLoader(mode="train", gpt_tokenizer=gpt_tokenizer, clip_model_type=args.clip_model_type,
                                     seq_len=args.seq_len, prefix_length=args.prefix_length)
    coco_data_val = CocoDataLoader(mode="val", gpt_tokenizer=gpt_tokenizer, clip_model_type=args.clip_model_type,
                                   seq_len=args.seq_len, prefix_length=args.prefix_length)

    train_loader = DataLoader(coco_data_train, batch_size=args.coco_batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(coco_data_val, batch_size=args.coco_batch_size, shuffle=True, num_workers=args.num_workers,
                            pin_memory=True, drop_last=True)

    trainer = CocoTrainer(args, train_loader, val_loader, device)
    trained_mapper_net = trainer.train()
    torch.save(trained_mapper_net.state_dict(), os.path.join(PROJECT_ROOT, "models", args.coco_trained_model_name))
    print("Model saved on path {}".format(PROJECT_ROOT + "/models"))


def main_episodic():
    """
    TaskDataLoader -- for episodic training / inference of multimodal few-shot tasks
    CocoTasksDataLoader -- for episodic training with coco
    """
    gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    experiment_id = "{}_{}-way_{}-shot".format(args.experiment_id, args.n_way, args.k_spt)
    meta = MetaTrainer(args, experiment_id, is_pretrained=False).to(device)

    params = list(filter(lambda p: p.requires_grad, meta.model.parameters()))
    params_summed = sum(p.numel() for p in params)
    print("Total num of params: {} ".format(params_summed))
    log_file_path = LOG_PATH + "log_{}.txt".format(experiment_id)
    write_data_to_txt(log_file_path, "Experiment ID: {} Date: {}, {}-way, {}-shot (support), {}-shot (query)"
                                     "Mapper: {}, Vision model: {}\n".format(experiment_id, datetime.datetime.now(),
                                                                             args.n_way, args.k_spt, args.k_qry,
                                                                             args.mapper_type, args.clip_model_type))

    data_loader = CocoTasksDataLoader(data_path=args.data_path, mode='train', n_way=args.n_way, k_shot=args.k_spt,
                                      k_query=args.k_qry, batchsz=10000, resize=args.img_size, seq_len=args.seq_len,
                                      repeats=args.repeats, tokenizer=gpt_tokenizer,  clip_model_type=args.clip_model_type,
                                      prefix_length=args.prefix_length)
    data_loader_test = CocoTasksDataLoader(data_path=args.data_path, mode='val', n_way=args.n_way, k_shot=args.k_spt,
                                           k_query=1, batchsz=100, resize=args.img_size, seq_len=args.seq_len,
                                           repeats=args.repeats, tokenizer=gpt_tokenizer,
                                           clip_model_type=args.clip_model_type, prefix_length=args.prefix_length)
    db = DataLoader(data_loader, batch_size=args.task_num, shuffle=True, num_workers=args.num_workers, pin_memory=True,
                    drop_last=True)

    for epoch in range(args.epoch // 10000):
        for step, (x_spt, y_spt, y_spt_mask, id_spt, x_qry, y_qry, y_qry_mask, qry_img_id) in enumerate(db):
            x_spt, y_spt, y_spt_mask, id_spt, x_qry, y_qry, y_qry_mask, qry_img_id = x_spt.to(device), y_spt.to(device), \
                                                                         y_spt_mask.to(device), id_spt, \
                                                                         x_qry.to(device), y_qry.to(device), \
                                                                        y_qry_mask.to(device), qry_img_id

            accs, losses = meta(x_spt, y_spt, y_spt_mask, id_spt, x_qry, y_qry, y_qry_mask, qry_img_id)

            if step % 100 == 0:
                print("------ Meta-training {}-way, {}-shot ({}-query) {}-mapper {}-prefix tokens------"
                      .format(args.n_way, args.k_spt, args.k_qry, args.mapper_type, args.prefix_length))
                print("Step: {} \tLosses: {}".format(step, losses))
                print("Step: {} \tTraining acc: {}\n".format(step, accs))
                write_data_to_txt(file_path=LOG_PATH + "log_{}.txt".format(experiment_id),
                                  data="Step: {} \tTraining acc: {} \n".format(step, accs))
                meta.save_mapper_model()

            if step % 400 == 0 and step != 0:  # evaluation
                db_test = DataLoader(data_loader_test, batch_size=1, shuffle=True, num_workers=args.num_workers,
                                     pin_memory=True)
                accs_all_test = []
                for x_spt, y_spt, y_spt_mask, x_qry, y_qry, y_qry_mask, y_qry_answer, qry_img_id in db_test:
                    x_spt, y_spt, y_spt_mask, x_qry, y_qry, y_qry_mask, y_qry_answer, qry_img_id = x_spt.squeeze(0).to(device), \
                                                                                                   y_spt.squeeze(0).to(device), \
                                                                                                   y_spt_mask.squeeze(0).to(device), \
                                                                                                   x_qry.to(device), \
                                                                                                   y_qry.to(device), \
                                                                                                   y_qry_mask.to(device), \
                                                                                                   y_qry_answer.to(device), \
                                                                                                   qry_img_id
                    accs = meta.finetunning(x_spt, y_spt, y_spt_mask, x_qry, y_qry, y_qry_mask, y_qry_answer, qry_img_id)
                    accs_all_test.append(accs)

                accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
                print("------ Meta-test {}-way, {}-shot ({}-query) ------" .format(args.n_way, args.k_spt, args.k_qry))
                print("Step: {} \tTest acc: {} \n".format(step, accs))
                write_data_to_txt(file_path=log_file_path, data="Step: {} \tTest acc: {} \n".format(step, accs))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--experiment_id', type=int, default=5000)
    argparser.add_argument('--data_root', type=str, default='real_name_mi')
    argparser.add_argument('--data_path', type=str, help='the path to datasets', default='')

    argparser.add_argument('--clip_model_type', type=str, default="ViT-B/32")
    argparser.add_argument('--epoch', type=int, help='epoch number', default=60000)
    argparser.add_argument('--n_way', type=int, help='n way', default=2)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=5)
    argparser.add_argument('--repeats', type=int, help='repeats for support set', default=0)
    argparser.add_argument('--img_size', type=int, help='img_size', default=224)
    argparser.add_argument('--seq_len', type=int, default=10)  # for padding batch
    argparser.add_argument('--prefix_length', type=int, default=4)
    argparser.add_argument('--mapper_type', type=str, default="ATT")
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for fine-tunning', default=5)
    argparser.add_argument('--num_workers', type=int, default=8)

    # COCO training / non-episodic image captioning
    argparser.add_argument('--coco_num_epochs', type=int, help='epoch number for COCO training', default=100)
    argparser.add_argument('--coco_batch_size', type=int, help='batch size for COCO training', default=64)
    argparser.add_argument('--coco_annotations_path', type=str, default=PATH + '/Datasets/coco/annotations/')
    argparser.add_argument('--early_stop_patience', type=int, help='#epochs w/o improvement', default=5)
    argparser.add_argument('--delta', type=float, help='min change in the monitored val loss', default=0.01)
    argparser.add_argument('--lr', type=float, help='LR for COCO training', default=3e-05)
    argparser.add_argument('--warm_up_steps', type=int, help='warm up steps', default=1000)
    argparser.add_argument('--coco_trained_model_name', type=str, default='coco_trained_model_v3.pt')
    args = argparser.parse_args()

    main_episodic()

