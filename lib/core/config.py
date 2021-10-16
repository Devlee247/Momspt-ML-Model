# -*- coding: utf-8 -*-

import argparse
import yaml

class DictAsMember(dict):
    def __getattr__(self, name):
        value = self[name]
        if isinstance(value, dict):
            value = DictAsMember(value)
        return value

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='cfg file path')

    args = parser.parse_args()
    cfg_file = args.cfg

    if args.cfg is not None:
        with open(cfg_file) as f:
            cfg = yaml.load(f, Loader= yaml.FullLoader)
        cfg = DictAsMember(cfg)
        return cfg, cfg_file
    else:
        print("please add --cfg argument!")
        exit(0)