"""
train
"""
import os
import paddle
import paddle.nn as nn
import paddle.optimizer as opt
import paddle.nn.functional as F
from paddle.optimizer.lr import CosineAnnealingDecay, LinearWarmup
from ..sample import uniform_sample, strict_fair_sample, Generator, WeightedSample
from ..utils import Dataset, seed_global
from ..super import super_bn as bn_module
from ..super import super_fc as fc_module
from ..super import super_conv as conv_module
from ..super.supernet import Supernet
from tqdm import tqdm

def train(
    name="",
    sample="fair",
    total_epoch=300,
    work_dir='./exp/',
    lr=0.1,
    weight_decay=5e-4,
    grad_clip=5,
    bnclass='BestBN',
    convclass='BestConv',
    fcclass='BestFC',
    sandwich=False,
    kd=False,
    n=1,
    train_arch=None,
    train_arch_prob=None
):
    space = [[ 4, 8, 12, 16]] * 7 + [[ 4, 8, 12, 16, 20, 24, 28, 32]] * 6 + [[ 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56,60, 64]] * 6
    max_network = [max(x) for x in space]
    min_network = [min(x) for x in space]
    os.makedirs(work_dir, exist_ok=True)
    model_path = (
        f'{name + "_" if name else ""}{sample}_sw{sandwich}_kd{kd}_n{n}_'
        + f"te{total_epoch}_lr{lr}_wd{weight_decay}_gc{grad_clip}"
        + f"_CLASS_{convclass}_{bnclass}_{fcclass}.pdparams"
    )
    model_path = os.path.join(work_dir, model_path)
    data = Dataset(cache="./data")
    train_loader = data.get_loader(128, "train", num_workers=8)

    supernet = Supernet(getattr(bn_module, bnclass), getattr(conv_module, convclass), getattr(fc_module, fcclass), space=space)
    
    optimizer = opt.Momentum(
        LinearWarmup(CosineAnnealingDecay(lr, total_epoch), 2000, 0.0, lr),
        momentum=0.9,
        parameters=supernet.parameters(),
        weight_decay=weight_decay,
        grad_clip=None if grad_clip <= 0 else nn.ClipGradByGlobalNorm(grad_clip),
    )

    supernet.train()
    anchor_archs = [max_network, min_network]
    if sample == 'uniform':
        next_arch = Generator(uniform_sample, space=space)
    elif sample == 'fair':
        next_arch = Generator(strict_fair_sample, space=space)
    elif sample == 'fixed':
        ws = WeightedSample(train_arch, train_arch_prob)
        next_arch = Generator(ws.sample, 1000)
         
    for e in range(total_epoch):
        with tqdm(train_loader, total=len(train_loader)) as t:
            for batch in t:
                # train n archs on one batch
                if sandwich:
                    # one big, one small and several middle
                    assert n >= len(anchor_archs), "you need to have at least anchor size of n to enable sandwich rule"
                    archs = anchor_archs + [next_arch() for _ in range(n - len(anchor_archs))]
                else: archs = [next_arch() for _ in range(n)]
                if kd: kd_distribution = None
                for arch in archs:
                    logit = supernet(batch[0], arch)
                    if not kd or kd_distribution is None:
                        if kd: kd_distribution = F.softmax(logit).detach()
                        loss = F.cross_entropy(logit, batch[1]) / n
                        loss.backward()
                    else:
                        # knowledge distillation
                        loss = F.cross_entropy(logit, kd_distribution, soft_label=True) / n
                        loss.backward()

                optimizer.step()
                optimizer.clear_grad()
                if (
                    optimizer._learning_rate.last_epoch
                    < optimizer._learning_rate.warmup_steps
                ):
                    optimizer._learning_rate.step()
                
        paddle.save(supernet.state_dict(), model_path.replace('.pdparams', f'-{e}.pdparams'))

        if optimizer._learning_rate.last_epoch >= optimizer._learning_rate.warmup_steps:
            optimizer._learning_rate.step()

if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--path", type=str, default="saved_models", help="path to save mid results and logs")
    parser.add_argument("--name", type=str, default="", help="exp name")
    parser.add_argument("--epoch", type=int, default=300, help="epoch to train supernet")
    parser.add_argument("--sample", type=str, choices=["uniform", "fair", "fixed"], help="sample strategy", default="fair")
    parser.add_argument('--train_arch', type=str)
    parser.add_argument('--train_arch_prob', type=str)
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0., help="learning rate")
    parser.add_argument("--clip", type=float, default=5, help="gradient clip")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--bnclass', type=str, default='BestBN')
    parser.add_argument('--convclass', type=str, default='BestConv')
    parser.add_argument('--fcclass', type=str, default='BestFC')
    parser.add_argument('--sandwich', action='store_true', help='force to use sandwith rule as sample methods')
    parser.add_argument('--kd', action='store_true', help='whether to use knowledge distillation')
    parser.add_argument('--n', type=int, default=1, help='the number of archs evaluated on the same data')

    args = parser.parse_args()

    seed_global(args.seed)

    for k, v in args.__dict__.items():
        print("{} : {}".format(k, v))

    paddle.set_device(f'gpu:{args.device}')

    train(
        name=args.name,
        sample=args.sample,
        total_epoch=args.epoch,
        work_dir=args.path,
        lr=args.lr if args.lr > 0 else -1,
        weight_decay=args.weight_decay,
        grad_clip=args.clip,
        bnclass=args.bnclass,
        convclass=args.convclass,
        fcclass=args.fcclass,
        sandwich=args.sandwich,
        kd=args.kd,
        n=args.n,
        train_arch=args.train_arch,
        train_arch_prob=args.train_arch_prob
    )
