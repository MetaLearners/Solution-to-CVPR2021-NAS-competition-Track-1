import paddle
import paddle.nn.functional as F
from tqdm import tqdm
import json
from ..super import super_bn as bn_module
from ..super import super_conv as conv_module
from ..super import super_fc as fc_module
from ..super.supernet import Supernet
from ..utils import str2arch, arch2str, Dataset

@paddle.no_grad()
def evaluate(path_to_pad, archs, convclass, bnclass, fcclass, bn_mode=3):
    model = Supernet(getattr(bn_module, bnclass), getattr(conv_module, convclass), getattr(fc_module, fcclass))
    model.set_state_dict(paddle.load(path_to_pad))
    dataset = Dataset(cache='./data')
    test_loader = dataset.get_loader(512, 'test', 4)
    if bn_mode in [1, 2]:
        cal_loader = dataset.get_loader(512, 'train', 4)
    else:
        cal_loader = test_loader

    results = {}
    for arch in tqdm(archs):
        # bn calibration
        if bn_mode in [1, 2]:
            model.train()
            for data in cal_loader:
                model(data[0], arch)
        
        model.eval()
        acc = 0
        loss = 0
        sizes = 0
        for data in test_loader:
            if bn_mode == 3:
                logit = model.inference(data[0], arch)
            else:
                logit = model(data[0], arch)
            sizes += len(logit)
            acc += (logit.argmax(1) == data[1]).numpy().sum()
            loss -= float(F.cross_entropy(logit, data[1], reduction='sum'))
        results[arch2str(arch)] = {
            'loss': loss / sizes,
            'acc': acc / sizes
        }
    return results

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--path_to_arch', type=str, default='./data/Track1_final_archs.json')
    parser.add_argument('--bnclass', type=str, default='BestBN')
    parser.add_argument('--convclass', type=str, default='BestConv')
    parser.add_argument('--fcclass', type=str, default='BestFC')
    parser.add_argument('--metric', type=str, default='loss')
    parser.add_argument('--output', type=str)
    parser.add_argument('--bn_mode', type=int, choices=[0,1,2,3], default=3)

    args = parser.parse_args()

    archs = json.load(open(args.path_to_arch))

    if isinstance(archs, dict):
        # submit version
        arch_now = [str2arch(archs[x]['arch']) for x in archs]
    else:
        arch_now = archs

    result = evaluate(args.path, arch_now, args.convclass, args.bnclass, args.fcclass, bn_mode=args.bn_mode)

    json.dump(result, open(args.output + '.raw', 'w'))

    if isinstance(archs, dict):
        maxs, mins = max([result[k][args.metric] for k in result]), min([result[k][args.metric] for k in result])
        for key in archs:
            archs[key]['acc'] = (result[archs[key]["arch"]][args.metric] - mins) / (maxs - mins)
        json.dump(archs, open(args.output, 'w'))

    else:
        import os
        os.system(f'mv {args.output + ".raw"} {args.output}')
