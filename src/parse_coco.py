import argparse
import json
import os
import pickle

import clip
import torch
from PIL import Image
from tqdm import tqdm

from utils import *

PATH = str(Path.cwd().parent.parent.parent) + '/Datasets/coco'  # Local machine Path

device = set_device()
if torch.cuda.is_available():
    print('Training on GPU!')
else:
    print('Training on CPU!')


def main(args, train_or_val):
    """
    Creates a pickle file of all img files and its captions
    """
    clip_model_type = args.clip_model_type
    coco_trained_model_id = args.coco_trained_model_id
    clip_model_name = clip_model_type.replace('/', '_')
    out_path = f"{PROJECT_ROOT}/data/coco/coco_split_{clip_model_name}_{train_or_val}_{coco_trained_model_id}.pkl"
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    with open(PATH + f"/annotations/captions_{train_or_val}2017.json", 'r') as f:
        data = json.load(f)['annotations']
    print("%0d captions loaded from json " % len(data))
    all_embeddings = []
    all_captions = []
    all_files = []

    for i in tqdm(range(len(data))):
        d = data[i]
        img_id = d["image_id"]
        filename = PATH + f"/images/train2017/{int(img_id):012d}.jpg"
        if not os.path.isfile(filename):
            filename = PATH + f"/images/val2017/{int(img_id):012d}.jpg"
        image = Image.open(filename)
        image = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = clip_model.encode_image(image).cpu()
        d["clip_embedding"] = i
        all_embeddings.append(prefix)
        all_captions.append(d)
        all_files.append(filename)
        if (i + 1) % 10000 == 0:
            with open(out_path, 'wb') as f:
                pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions,
                             "image_files": all_files}, f)

    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions,
                     "image_files": all_files}, f)

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))


def get_coco_categories_to_img_caps(mode):
    """
    Returns all COCO training samples: image, caption, category.
    annotations: image_id -> category_id
    categories: id -> name
    output: category_id /name -> image/caption/ pair
    """
    with open(PATH + f"/annotations/instances_{mode}2017.json") as json_file:
        data = json.load(json_file)
    images = data['images']
    annotations = data['annotations']
    categories = data['categories']
    cat_id_to_cat_name = get_cat_id_to_cat_name(categories)
    img_id_to_cat_name = get_img_id_to_cat_name(annotations)

    with open(PATH + f"/annotations/captions_{mode}2017.json") as json_file:
        captions_data = json.load(json_file)
    captions = captions_data['annotations']
    img_id_to_caption = get_img_id_to_caption(captions)
    cat_to_img_id = get_category_to_img_id_pairs(images, cat_id_to_cat_name, img_id_to_caption, img_id_to_cat_name)

    return cat_to_img_id


def get_category_to_img_id_pairs(images, cat_id_to_cat_name, img_id_to_caption, img_id_to_cat_name):
    out_dict = {}
    for a in images:
        img_id = a['id']
        if img_id in img_id_to_cat_name.keys():
            category_id = img_id_to_cat_name[img_id]
            category_name = cat_id_to_cat_name[category_id]
            captions = img_id_to_caption[img_id]

            images_list = []
            if category_name in out_dict.keys():
                images_list = out_dict[category_name]

            for caption in captions:
                images_list.append((caption, img_id))

            out_dict[category_name] = images_list

    return out_dict


def get_cat_id_to_cat_name(categories):
    out_dict = {}
    for a in categories:
        category_id = a['id']
        name = a['name']
        out_dict[category_id] = name

    return out_dict


def get_img_id_to_cat_name(annotations):
    out_dict = {}
    for a in annotations:
        category_id = a['category_id']
        img_id = a['image_id']
        out_dict[img_id] = category_id

    return out_dict


def get_img_id_to_caption(captions):
    out_dict = {}
    for a in captions:
        caption = a['caption']
        img_id = a['image_id']
        if img_id in out_dict.keys():
            captions = out_dict[img_id]
            captions.append(caption)
        else:
            out_dict[img_id] = [caption]

    return out_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'ViT-B/32'))
    parser.add_argument('--coco_trained_model_id', type=int, default=2)
    args = parser.parse_args()
    main(args, "train")
    main(args, "val")
