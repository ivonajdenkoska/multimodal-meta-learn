import json
import os
import pickle
import random
from pathlib import Path

import clip
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter

from parse_coco import get_coco_categories_to_img_caps
from utils import set_device

PATH = str(Path.cwd().parent)
DATA_PATH = str(Path.cwd().parent.parent.parent) + '/Datasets'  # Local machine Path
PROJECT_ROOT = str(Path.cwd().parent)  # project path


class TaskDataLoader(Dataset):
    """
    This is a DataLoader for loading of the few-shot tasks (for inference time)
    NOTICE: meta-learning is different from general supervised learning, especially the concept of batch and set.
    batch: contains several sets / tasks
    sets: conains n_way * k_shot for meta-train set, n_way * k_query for meta-test set.

    Datasets available on https://fh295.github.io/frozen.html
    """

    def __init__(self, root, mode, batchsz, n_way, k_shot, k_query, resize, seq_len, repeats, tokenizer,
                 clip_model_type, prefix_length, startidx=0):
        self.batchsz = batchsz  # batch of set, not batch of imgs
        self.n_way = n_way  # n-way
        self.k_shot = k_shot  # k-shot
        self.k_query = k_query  # for evaluation
        self.repeats = repeats
        self.setsz = self.n_way * self.k_shot if self.repeats == 0 else self.n_way * self.k_shot * (self.repeats + 1)
        self.querysz = self.n_way * self.k_query  # number of samples per set for evaluation
        self.resize = resize  # resize to
        self.seq_len = seq_len  # sentence seq length
        self.prefix_length = prefix_length
        self.startidx = startidx  # index label not from 0, but from startidx
        self.device = set_device()
        print('shuffle DB: %s, b:%d, %d-way, %d-shot, %d-query, %d-repeats, resize:%d' % (mode, batchsz, n_way, k_shot,
                                                                                          k_query, repeats, resize))

        self.gpt_tokenizer = tokenizer
        self.clip, self.clip_preprocess = clip.load(clip_model_type, device=self.device, jit=False)

        self.path = os.path.join(DATA_PATH, root)  # image path
        with open(PATH + "/data/meta_{}_{}_shots_{}_ways.json".format(mode, self.k_shot, self.n_way)) as f:
            jsonf = json.load(f)
        json_data = self.load_JSON(jsonf, self.n_way, self.k_shot)

        self.data = []
        self.img2caption = {}

        for i, (caption, images) in enumerate(json_data.items()):
            self.data.append(images)
            for image in images:
                self.img2caption[image] = caption

        self.cls_num = len(self.data)
        self.create_batch(self.batchsz)

    def load_JSON(self, jsonf, num_ways, num_shots):
        """
        Returns a dictionary
        - key == caption
        - value == array of all images associated to the caption
        """
        full_dict = {}
        for q in jsonf:
            for i in range(1, num_shots * num_ways + 1):
                if 'image_{}'.format(i) in q.keys():
                    image = q['image_{}'.format(i)]
                    caption = q['caption_{}'.format(i)]
                    if caption in full_dict.keys():
                        full_dict[caption].append(image)
                    else:
                        full_dict[caption] = [image]

        return full_dict

    def create_batch(self, batchsz):
        """
        create batch for meta-learning.
        ×episode× here means batch, and it means how many sets we want to retain.
        :param episodes: batch size
        :return:
        """
        self.support_x_batch = []  # support set batch
        self.query_x_batch = []  # query set batch
        # Creating of tasks; batchsz is the num. of iterations when sampling from the task distribution
        for b in range(batchsz):  # for each batch
            # 1.select n_way classes randomly
            selected_cls = np.random.choice(self.cls_num, self.n_way, replace=False)  # no duplicate
            np.random.shuffle(selected_cls)
            support_x = []
            query_x = []
            for cls in selected_cls:
                # 2. select k_shot + k_query for each class
                selected_imgs_idx = np.random.choice(len(self.data[cls]), self.k_shot + self.k_query, False)
                np.random.shuffle(selected_imgs_idx)
                indexDtrain = np.array(selected_imgs_idx[:self.k_shot])  # idx for Dtrain
                indexDtest = np.array(selected_imgs_idx[self.k_shot:])  # idx for Dtest
                support_x.append(
                    np.array(self.data[cls])[indexDtrain].tolist())  # get all images filename for current Dtrain
                query_x.append(np.array(self.data[cls])[indexDtest].tolist())
                if self.repeats > 0:
                    for i in range(self.repeats):
                        support_x.append(np.array(self.data[cls])[indexDtrain].tolist())

            # shuffle the corresponding relation between support set and query set
            random.shuffle(support_x)
            random.shuffle(query_x)

            self.support_x_batch.append(support_x)  # append set to current sets
            self.query_x_batch.append(query_x)  # append sets to current sets

    def __getitem__(self, index):
        support_x = torch.FloatTensor(self.setsz, 3, self.resize, self.resize)
        support_y = []
        support_y_mask = []
        query_x = torch.FloatTensor(self.querysz, 3, self.resize, self.resize)
        query_y = []
        query_y_mask = []

        # image path files
        flatten_support_x = [os.path.join(self.path, item)
                             for sublist in self.support_x_batch[index] for item in sublist]
        flatten_query_x = [os.path.join(self.path, item)
                           for sublist in self.query_x_batch[index] for item in sublist]

        for sublist in self.support_x_batch[index]:
            for item in sublist:
                caption = "{}".format(self.img2caption[item].lower())
                caption_tokenized = self.gpt_tokenizer(caption, return_tensors="pt")['input_ids']
                caption_padded, mask = pad_tokens(caption_tokenized, self.seq_len, self.prefix_length,
                                                  self.gpt_tokenizer.eos_token_id)
                support_y.append(caption_padded)
                support_y_mask.append(mask)

        support_y = torch.stack(support_y).squeeze(1)
        support_y_mask = torch.stack(support_y_mask).squeeze(1)

        for sublist in self.query_x_batch[index]:
            for item in sublist:
                caption = "{}".format(self.img2caption[item].lower())
                caption_tokenized = self.gpt_tokenizer(caption, return_tensors="pt")['input_ids']
                caption_padded, mask = pad_tokens(caption_tokenized, self.seq_len, self.prefix_length,
                                                  self.gpt_tokenizer.eos_token_id)
                query_y.append(caption_padded)
                query_y_mask.append(mask)

        query_y = torch.stack(query_y).squeeze(1)
        query_y_mask = torch.stack(query_y_mask).squeeze(1)

        # Reading of images:
        for i, path in enumerate(flatten_support_x):
            image = self.clip_preprocess(Image.open(path))
            support_x[i] = image

        for i, path in enumerate(flatten_query_x):
            image = self.clip_preprocess(Image.open(path))
            query_x[i] = image

        return support_x, support_y, support_y_mask, query_x, query_y, query_y_mask

    def __len__(self):
        return self.batchsz


class MetaTestTaskDataLoader(Dataset):
    """
        DataLoader for loading the full few-shot datasets for meta-test
    """

    def __init__(self, root, mode, batchsz, n_way, k_shot, k_query, resize, seq_len, repeats, tokenizer,
                 clip_model_type, prefix_length):
        self.batchsz = batchsz  # batch of set, not batch of imgs
        self.n_way = n_way  # n-way
        self.k_shot = k_shot  # k-shot
        self.k_query = k_query  # for evaluation
        self.repeats = repeats
        self.setsz = self.n_way * self.k_shot if self.repeats == 0 else self.n_way * self.k_shot * (self.repeats + 1)
        self.querysz = self.n_way * self.k_query  # number of samples per set for evaluation
        self.seq_len = seq_len  # sentence seq length
        self.prefix_length = prefix_length
        self.device = set_device()
        print('shuffle DB: %s, b:%d, %d-way, %d-shot, %d-query, %d-repeats, resize:%d' % (mode, batchsz, n_way, k_shot,
                                                                                          k_query, repeats, resize))

        self.gpt_tokenizer = tokenizer
        self.clip, self.clip_preprocess = clip.load(clip_model_type, device=self.device, jit=False)

        self.root = root
        self.path = os.path.join(DATA_PATH, root)  # image path
        with open(DATA_PATH + f"/{root}/{root}_shots_{self.k_shot}_ways_{self.n_way}_all_questions.json") as f:
            self.jsonf = json.load(f)

    def __getitem__(self, index):
        task = self.jsonf[index]
        support_x, support_y, support_y_mask = [], [], []
        question = f"{self.jsonf[index]['question']}"

        for i in range(1, self.k_shot * self.n_way + 1):
            image_file = task['image_{}'.format(i)]
            image_path = DATA_PATH + f"/{self.root}/{image_file}"
            image = self.clip_preprocess(Image.open(image_path))
            support_x.append(image)

            caption = task['caption_{}'.format(i)]
            caption_tokenized = self.gpt_tokenizer(caption, return_tensors="pt")['input_ids']
            caption_padded, mask = pad_tokens(caption_tokenized, self.seq_len, self.prefix_length,
                                              self.gpt_tokenizer.eos_token_id)
            support_y.append(caption_padded)
            support_y_mask.append(mask)

        support_x = torch.stack(support_x).squeeze(1)
        support_y = torch.stack(support_y).squeeze(1)
        support_y_mask = torch.stack(support_y_mask).squeeze(1)

        question_img = self.jsonf[index]['question_image']
        image_path = DATA_PATH + f"/{self.root}/{question_img}"
        question_image = self.clip_preprocess(Image.open(image_path))

        question_tokenized = self.gpt_tokenizer(question, return_tensors="pt")['input_ids']
        question_full, question_mask = pad_tokens(question_tokenized, self.seq_len, self.prefix_length,
                                                  self.gpt_tokenizer.eos_token_id)
        answer = self.gpt_tokenizer(self.jsonf[index]['answer'], return_tensors="pt")['input_ids']
        query_answer, _ = pad_tokens(answer, self.seq_len, self.prefix_length, self.gpt_tokenizer.eos_token_id)

        return support_x, support_y, support_y_mask, question_image, question_full, question_mask, query_answer, question_img

    def __len__(self):
        return len(self.jsonf)


class CocoTasksDataLoader(Dataset):
    """
    This is DataLoader for episodic training on COCO dataset
    NOTICE: meta-learning is different from general supervised learning, especially the concept of batch and set.
    batch: contains several sets / tasks
    sets: conains n_way * k_shot for meta-train set, n_way * k_query for meta-test set.
    """

    def __init__(self, data_path, mode, batchsz, n_way, k_shot, k_query, resize, seq_len, repeats, tokenizer,
                 clip_model_type, prefix_length, startidx=0):
        self.batchsz = batchsz  # batch of set, not batch of imgs
        self.n_way = n_way  # n-way
        self.k_shot = k_shot  # k-shot
        self.k_query = k_query  # for evaluation
        self.repeats = repeats
        self.setsz = self.n_way * self.k_shot if self.repeats == 0 else self.n_way * self.k_shot * (self.repeats + 1)
        self.querysz = self.n_way * self.k_query  # number of samples per set for evaluation
        self.resize = resize  # resize to
        self.seq_len = seq_len  # sentence seq length
        self.prefix_length = prefix_length
        self.startidx = startidx  # index label not from 0, but from startidx
        self.device = set_device()
        print('shuffle DB: %s, b:%d, %d-way, %d-shot, %d-query, %d-repeats, resize:%d' % (mode, batchsz, n_way, k_shot,
                                                                                          k_query, repeats, resize))

        self.gpt_tokenizer = tokenizer
        self.clip, self.clip_preprocess = clip.load(clip_model_type, device=self.device, jit=False)

        self.path = data_path  # image path
        self.mode = mode
        json_data = get_coco_categories_to_img_caps(mode)

        self.data = []
        self.img2caption = {}

        for i, (category_name, cap_imgs) in enumerate(json_data.items()):
            self.data.append(cap_imgs)
            for caption, image in cap_imgs:
                self.img2caption[image] = caption

        self.cls_num = len(self.data)
        self.create_batch(self.batchsz)

    def create_batch(self, batchsz):
        """
        create batch for meta-learning.
        ×episode× here means batch, and it means how many sets we want to retain.
        :param episodes: batch size
        :return:
        """
        self.support_x_batch = []  # support set batch
        self.query_x_batch = []  # query set batch
        # Creating of tasks; batchsz is the num. of iterations when sampling from the task distribution
        for b in range(batchsz):  # for each batch
            # 1.select n_way classes randomly
            selected_cls = np.random.choice(self.cls_num, self.n_way, replace=False)  # no duplicate
            np.random.shuffle(selected_cls)
            support_x = []
            query_x = []
            for cls in selected_cls:
                # 2. select k_shot + k_query for each class
                selected_imgs_idx = np.random.choice(len(self.data[cls]), self.k_shot + self.k_query, False)
                np.random.shuffle(selected_imgs_idx)
                indexDtrain = np.array(selected_imgs_idx[:self.k_shot])  # idx for Dtrain
                indexDtest = np.array(selected_imgs_idx[self.k_shot:])  # idx for Dtest
                support_x.append(
                    np.array(self.data[cls])[indexDtrain].tolist())  # get all images filename for current Dtrain
                query_x.append(np.array(self.data[cls])[indexDtest].tolist())
                if self.repeats > 0:
                    for i in range(self.repeats):
                        support_x.append(np.array(self.data[cls])[indexDtrain].tolist())

            # shuffle the corresponding relation between support set and query set
            random.shuffle(support_x)
            random.shuffle(query_x)

            self.support_x_batch.append(support_x)  # append set to current sets
            self.query_x_batch.append(query_x)  # append sets to current sets

    def __getitem__(self, index):
        """
        index means index of sets, 0<= index <= batchsz-1
        :param index:
        :return:
        """
        support_x = torch.FloatTensor(self.setsz, 3, self.resize, self.resize)
        support_y = []
        support_y_mask = []
        query_x = torch.FloatTensor(self.querysz, 3, self.resize, self.resize)
        query_y = []
        query_y_mask = []

        # image path files  f"/coco/images/{self.mode}2017/{int(img_id):012d}.jpg"
        flatten_support_x = [f"{self.path}/coco/images/{self.mode}2017/{int(img_id):012d}.jpg"
                             for sublist in self.support_x_batch[index] for _, img_id in sublist]
        flatten_query_x = [f"{self.path}/coco/images/{self.mode}2017/{int(img_id):012d}.jpg"
                           for sublist in self.query_x_batch[index] for _, img_id in sublist]

        for sublist in self.support_x_batch[index]:
            for caption, img_id in sublist:
                caption = caption.lower()
                caption_tokenized = self.gpt_tokenizer(caption, return_tensors="pt")['input_ids']
                caption_padded, mask = pad_tokens(caption_tokenized, self.seq_len, self.prefix_length,
                                                  self.gpt_tokenizer.eos_token_id)
                support_y.append(caption_padded)
                support_y_mask.append(mask)
        support_y = torch.stack(support_y).squeeze(1)
        support_y_mask = torch.stack(support_y_mask).squeeze(1)

        for sublist in self.query_x_batch[index]:
            for caption, img_id in sublist:
                caption = caption.lower()
                caption_tokenized = self.gpt_tokenizer(caption, return_tensors="pt")['input_ids']
                caption_padded, mask = pad_tokens(caption_tokenized, self.seq_len, self.prefix_length,
                                                  self.gpt_tokenizer.eos_token_id)
                query_y.append(caption_padded)
                query_y_mask.append(mask)

        query_y = torch.stack(query_y).squeeze(1)
        query_y_mask = torch.stack(query_y_mask).squeeze(1)

        # Reading of images:
        for i, path in enumerate(flatten_support_x):
            image = self.clip_preprocess(Image.open(path))
            support_x[i] = image

        for i, path in enumerate(flatten_query_x):
            image = self.clip_preprocess(Image.open(path))
            query_x[i] = image

        return support_x, support_y, support_y_mask, flatten_support_x, query_x, query_y, query_y_mask, flatten_query_x

    def __len__(self):
        return self.batchsz


class CocoDataLoader(Dataset):
    def __init__(self, mode, gpt_tokenizer, clip_model_type, seq_len, prefix_length):
        self.gpt_tokenizer = gpt_tokenizer
        self.seq_len = seq_len
        self.prefix_length = prefix_length
        self.mode = mode
        self.device = set_device()
        self.clip, self.clip_preprocess = clip.load(clip_model_type, device=self.device, jit=False)
        data_path = PROJECT_ROOT + f"/data/coco/coco_split_ViT-B_32_{mode}.pkl"
        with open(f"{data_path}", 'rb') as f:
            data = pickle.load(f)
        self.clip_embeddings = data['clip_embedding']
        self.captions = data['captions']

        self.img_transforms = ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)

    def __getitem__(self, index):
        clip_encoding = self.clip_embeddings[index]
        caption = self.captions[index]['caption']
        img_id = self.captions[index]['image_id']
        img_file_name = DATA_PATH + f"/coco/images/{self.mode}2017/{int(img_id):012d}.jpg"
        image_read = Image.open(img_file_name)
        image = self.img_transforms(image_read) if self.mode == "train" else image_read
        image_preprocessed = self.clip_preprocess(image)

        caption_tokenized = self.gpt_tokenizer(caption, return_tensors="pt")['input_ids']
        caption_padded, cap_mask = pad_tokens(caption_tokenized, self.seq_len, self.prefix_length,
                                              self.gpt_tokenizer.eos_token_id)
        return image_preprocessed, clip_encoding, caption_padded, cap_mask

    def __len__(self):
        return len(self.clip_embeddings)


def pad_tokens(tokens, seq_len, prefix_length, eos_token_id):
    tokens = tokens.squeeze(0)
    padding = seq_len - tokens.shape[0]
    if padding > 0:
        tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
    elif padding < 0:
        tokens = tokens[:seq_len]
    mask = tokens.ge(0)  # mask is zero where we out of sequence
    tokens[~mask] = eos_token_id
    mask = mask.float()
    mask = torch.cat((torch.ones(prefix_length), mask), dim=0)  # adding prefix mask
    return tokens, mask
