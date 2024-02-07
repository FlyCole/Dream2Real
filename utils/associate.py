import argparse
import pathlib
import pdb
import sys
import os
import numpy as np


def read_file_list(filename):
    file = open(filename)
    data = file.read()
    lines = data.replace(",", " ").replace("\t", " ").split("\n")
    list = [[v.strip() for v in line.split(" ") if v.strip() != ""] for line in lines if
            len(line) > 0 and line[0] != "#"]
    list = [(float(l[0]), 0) for l in list]
    return dict(list)


def associate(data_dir):
    associate_file = os.path.join(data_dir, "associate_index.txt")
    if os.path.exists(associate_file):
        associate_index = read_file_list(associate_file)
        associate_list = list(associate_index)
        associate_list = list(map(int, associate_list))
    else:
        first_list = read_file_list(os.path.join(data_dir, "rgb_timestamps.txt"))
        second_list = read_file_list(os.path.join(data_dir, "seg_timestamps.txt"))

        offset = 0.0
        max_difference = 1e9  # 1 second

        first_keys = list(first_list)
        second_keys = list(second_list)

        associate_list = []
        for i in range(len(first_keys)):
            min_diff = max_difference
            min_index = None
            for j in range(len(second_keys)):
                if i == 0:
                    if first_keys[i] > second_keys[j]:
                        continue
                diff = abs(first_keys[i] - (second_keys[j] + offset))
                if diff < min_diff:
                    min_diff = diff
                    min_index = j

            associate_list.append(min_index)
        assert len(associate_list) == len(first_keys)

    return associate_list


if __name__ == '__main__':
    associate_list = associate()
    print(associate_list)


