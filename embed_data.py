# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)

import os
import sys
import argparse
import numpy as np
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import src.config as config
from src.embed import get_embedder
from src.utils import Print, set_seeds, set_output, check_args

parser = argparse.ArgumentParser('Embed Data')
parser.add_argument('--data-path',  help='path for data configuration file')
parser.add_argument('--model-config', help='path for model configuration file')


def main():
    args = vars(parser.parse_args())
    check_args(args)
    set_seeds(2021)
    model_cfg = config.ModelConfig(args["model_config"])
    args["output_path"] = "%s/%s/" % (args["data_path"], model_cfg.embedder)
    output, save_prefix = set_output(args, "embed_data_log")

    embedder = get_embedder(model_cfg.embedder)
    for file in sorted(os.listdir(args["data_path"] + "/FASTA")):
        if not file.endswith("fasta"): continue
        data_idx = os.path.splitext(file)[0]
        os.makedirs(save_prefix + "/%s/" % (data_idx), exist_ok=True)

        FILE = open(args["data_path"] + "/FASTA/%s.fasta" % data_idx, "r")
        lines = FILE.readlines()
        FILE.close()

        start = Print('start embedding %s' % data_idx, output)
        for i, line in enumerate(lines):
            if line.startswith(">"): continue
            elif not os.path.exists(save_prefix + "/%s/%d.npy" % (data_idx, i // 2)):
                seq = line.strip().upper()
                if model_cfg.embedder == "ESM": seq = seq.replace("J", "X")

                e = embedder.embed(seq)
                if model_cfg.embedder == "SeqVec":   e = np.sum(e, axis=0)
                elif model_cfg.embedder == "UniRep": e = e[1:]
                np.save(save_prefix + "/%s/%d.npy" % (data_idx, i // 2), e)

            if (i // 2) % 10 == 0:
                print('# {} {:.1%}'.format(data_idx, (i // 2) / ((len(lines)-1) // 2)), end='\r', file=sys.stderr)
        print(' ' * 15, end='\r', file=sys.stderr)
        end = Print('end embedding %s' % data_idx, output)
        Print(" ".join(['elapsed time:', str(end - start)]), output, newline=True)

    if not output == sys.stdout: output.close()


if __name__ == '__main__':
    main()
