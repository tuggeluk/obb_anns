"""Cluster bboxes.

Analysis of bounding box sizes present in dataset,
used inform tuning of the hyperparameters of the detectors.

Author:
    Lukas Tuggener <tugg@zhaw.ch>

Created On:
    October 12, 2020
"""
from os.path import join
from obb_anns import OBBAnns
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def parse_args():
    parser = ArgumentParser(description='runs the obb_anns.py file')
    parser.add_argument('ROOT', type=str,
                        help='path to the root of the dataset directory')
    parser.add_argument('ANNS', type=str,
                        help='name of the annotation file to use')
    parser.add_argument('PROPOSAL', type=str, nargs='?',
                        help='name of the proposals json')
    return parser.parse_args()



def extract_bbox_list():
    all_bboxes = []
    return all_bboxes


if __name__ == '__main__':
    args = parse_args()

    a = OBBAnns(join(args.ROOT, args.ANNS))
    a.load_annotations()
    a.set_annotation_set_filter(['deepscores'])

    np_annotations = np.stack(a.ann_info['a_bbox'])
    height = np_annotations[:, 3] - np_annotations[:, 1]
    width = np_annotations[:, 2] - np_annotations[:, 0]
    aspect_ratio = height / width
    area = a.ann_info['area']

    d = {'height': height, 'width': width, 'aspect_ratio': aspect_ratio, 'area': area}
    df = pd.DataFrame(data=d)

    df_filtered = df.loc[(df[['height', 'width']] != 0).all(axis=1)]

    fig, ax = plt.subplots()
    hex_plt = ax.hexbin(df_filtered['height'], df_filtered['width'],bins="log", xscale="log",yscale="log",mincnt=5)
    plt.gca().set_aspect('equal', adjustable='box')
    ax.set_ylabel("width")
    ax.set_xlabel("height")
    cb = fig.colorbar(hex_plt, ax=ax)
    cb.set_label('log10(N)')
    plt.show()

    ax = sns.displot(x=np.log10(df_filtered['height']), kde=True)
    ax.set(xlabel="height (log10)", ylabel="Count")
    plt.show()

    ax = sns.displot(x=np.log10(df_filtered['width']), kde=True)
    ax.set(xlabel="width (log10)", ylabel="Count")
    plt.show()