import torch
import torch.nn as nn
from quant import *
import multiprocessing
import argparse
import os
import random
import numpy as np
import time
import hubconf
from data.imagenet import build_imagenet_data


def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        #print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


@torch.no_grad()
def validate_model(val_loader, model, device=None, print_freq=100):
    if device is None:
        device = next(model.parameters()).device
    else:
        model.to(device)
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (images, target) in enumerate(val_loader):
        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        #if i % print_freq == 0:
        #    progress.display(i)
    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg


def get_train_samples(train_loader, num_samples):
    train_data = []
    for batch in train_loader:
        train_data.append(batch[0])
        if len(train_data) * batch[0].size(0) >= num_samples:
            break
    return torch.cat(train_data, dim=0)[:num_samples]

def recon_model(model, partitions, **kwargs):
    """
    Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
    """
    for name, module in model.named_modules():
        if isinstance(module, QuantModule) and not module.ignore_reconstruction:
            if "layer" not in name:
                print('Reconstruction for layer {}'.format(name))
                layer_reconstruction(model, module, **kwargs)
            for partition in partitions:
                if partition[-1] == name:
                    modules = []
                    for name, module in model.named_modules():
                        if name in partition:
                            modules.append((name, module))
                    print("Reconstruction for partition", partition)
                    partition_reconstruction(model, modules, **kwargs)


def evaluate(args, partitions):
    print("EVALUATE", partitions)
    seed_all(args.seed)
    # build imagenet data loader
    train_loader, test_loader = build_imagenet_data(batch_size=args.batch_size, workers=args.workers,
                                                    data_path=args.data_path)

    # load model
    cnn = eval('hubconf.{}(pretrained=True)'.format(args.arch))
    cnn.cuda()
    cnn.eval()
    # build quantization parameters
    wq_params = {'n_bits': args.n_bits_w, 'channel_wise': args.channel_wise, 'scale_method': 'mse'}
    aq_params = {'n_bits': args.n_bits_a, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': args.act_quant}
    qnn = QuantModel(model=cnn, weight_quant_params=wq_params, act_quant_params=aq_params)
    qnn.cuda()
    qnn.eval()
    if not args.disable_8bit_head_stem:
        print('Setting the first and the last layer to 8-bit')
        qnn.set_first_last_layer_to_8bit()

    cali_data = get_train_samples(train_loader, num_samples=args.num_samples)
    device = next(qnn.parameters()).device

    # Initialize weight quantization parameters
    qnn.set_quant_state(True, False)
    _ = qnn(cali_data[:64].to(device))

    if args.test_before_calibration:
        print('Quantized accuracy before brecq: {}'.format(validate_model(test_loader, qnn)))

    # Kwargs for weight rounding calibration
    kwargs = dict(cali_data=cali_data, iters=args.iters_w, weight=args.weight, asym=True,
                  b_range=(args.b_start, args.b_end), warmup=args.warmup, act_quant=False, opt_mode='mse')


    # Start calibration
    recon_model(qnn, partitions, **kwargs)
    qnn.set_quant_state(weight_quant=True, act_quant=False)
    score = validate_model(test_loader, qnn)
    print('Weight quantization accuracy: {}'.format(score))

    if args.act_quant:
        # Initialize activation quantization parameters
        qnn.set_quant_state(True, True)
        with torch.no_grad():
            _ = qnn(cali_data[:64].to(device))
        # Disable output quantization because network output
        # does not get involved in further computation
        qnn.disable_network_output_quantization()
        # Kwargs for activation rounding calibration
        kwargs = dict(cali_data=cali_data, iters=args.iters_a, act_quant=True, opt_mode='mse', lr=args.lr, p=args.p)
        recon_model(qnn, partitions, **kwargs)
        qnn.set_quant_state(weight_quant=True, act_quant=True)

        score = validate_model(test_loader, qnn)
        print('Full quantization (W{}A{}) accuracy: {}'.format(args.n_bits_w, args.n_bits_a, score))

    return score

def _evaluate(solution, names, args):
    partitions = []
    partition = [names[0]]
    for switch, name in zip(solution, names[1:]):
        if switch:
            partitions.append(partition)
            partition = [name]
        else:
            partition.append(name)
    partitions.append(partition)
    score = evaluate(args, partitions)
    return score

class GA(object):

    def __init__(self, args, names, populations, generations, num_workers=0):
        self.populations = populations
        self.replace_nums = populations // 2
        self.generations = generations
        self.names = names
        self.args = args
        self.num_workers = num_workers

    def init_chromosome(self):
        solutions = []
        for _ in range(self.populations):
            solution = [random.randint(0, 1) for _ in range(len(self.names) - 1)]
            solutions.append(solution)
        self.solutions = solutions

        if self.num_workers > 1:
            pool = multiprocessing.Pool(processes=self.num_workers)
            result = [pool.apply_async(_evaluate, [solution, self.names, self.args]) for solution in solutions]
            pool.close()
            pool.join()
            self.scores = [r.get() for r in result]
        else:
            self.scores = [self.evaluate(solution) for solution in solutions]

    def evaluate(self, solution):
        partitions = []
        partition = [self.names[0]]
        for switch, name in zip(solution, self.names[1:]):
            if switch:
                partitions.append(partition)
                partition = [name]
            else:
                partition.append(name)
        partitions.append(partition)
        score = evaluate(self.args, partitions)

        return score

    def selection(self, rank_weight_max=5, rank_weight_min=1):
        def rank_based_selection(probs):
            a_idx, b_idx = np.random.choice(len(probs), size=2, replace=False, p=probs)
            return a_idx, b_idx

        idx = []
        selection_weights = np.linspace(
            rank_weight_min, rank_weight_max, num=len(self.scores),
        )
        selection_probs = selection_weights / selection_weights.sum()
        for _ in range(self.replace_nums):
            a, b = rank_based_selection(selection_probs)
            idx.append((a, b))
        return idx

    def crossover(self, parents):
        childs = []
        for a, b in parents:
            a, b = self.solutions[a], self.solutions[b]
            r_idx = random.randint(0, len(a)-1)
            child = a[:r_idx] + b[r_idx:]
            childs.append(child)
        return childs

    def mutation(self, childs):
        for child in childs:
            a, b = random.sample(range(len(self.names) - 2), 2)
            child[a], child[b] = child[b], child[a]
        return childs

    def replace(self, childs):

        if self.num_workers > 1:
            pool = multiprocessing.Pool(processes=self.num_workers)
            result = [pool.apply_async(_evaluate, [child, self.names, self.args]) for child in childs]
            pool.close()
            pool.join()
            child_scores = [r.get() for r in result]
        else:
            child_scores = [self.evaluate(child) for child in childs]

        for idx, child in enumerate(childs):
            self.solutions[idx] = child
            self.scores[idx] = child_scores[idx]

    def _log(self, generation):
        print("GEN [{}] BEST SCORE {}".format(generation, self.scores[-1]))

    def run(self):
        self.init_chromosome()

        for generation in range(self.generations):
            tuples = list(zip(self.scores, self.solutions))
            tuples = sorted(tuples, key=lambda x: -x[0])
            self.scores = [x[0] for x in tuples]
            self.solutions = [x[1] for x in tuples]
            self._log(generation)

            parents = self.selection()
            childs = self.crossover(parents)
            childs = self.mutation(childs)
            self.replace(childs)
            self._log(generation)

        tuples = list(zip(self.scores, self.solutions))
        tuples = sorted(tuples, key=lambda x: x[0])
        self.scores = [x[0] for x in tuples]
        self.solutions = [x[1] for x in tuples]

        best_score = self.scores[-1]
        best_solution = self.solutions[-1]
        return best_score, best_solution


def run_ga(args):
    cnn = eval('hubconf.{}(pretrained=True)'.format(args.arch))
    wq_params = {'n_bits': args.n_bits_w, 'channel_wise': args.channel_wise, 'scale_method': 'mse'}
    aq_params = {'n_bits': args.n_bits_a, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': args.act_quant}
    model = QuantModel(model=cnn, weight_quant_params=wq_params, act_quant_params=aq_params)
    names = []
    for name, module in model.named_modules():
        if isinstance(module, QuantModule) and not module.ignore_reconstruction:
            if "layer" in name:
                names.append(name)


    ga = GA(args, names, args.populations, args.generations, args.num_workers)
    ga.run()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='running parameters',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # general parameters for data and model
    parser.add_argument('--seed', default=1005, type=int, help='random seed for results reproduction')
    parser.add_argument('--arch', default='resnet18', type=str, help='dataset name',
                        choices=['resnet18', 'resnet50', 'mobilenetv2', 'regnetx_600m', 'regnetx_3200m', 'mnasnet'])
    parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size for data loader')
    parser.add_argument('--workers', default=4, type=int, help='number of workers for data loader')
    parser.add_argument('--data_path', default='', type=str, help='path to ImageNet data', required=True)

    # quantization parameters
    parser.add_argument('--n_bits_w', default=4, type=int, help='bitwidth for weight quantization')
    parser.add_argument('--channel_wise', action='store_true', help='apply channel_wise quantization for weights')
    parser.add_argument('--n_bits_a', default=4, type=int, help='bitwidth for activation quantization')
    parser.add_argument('--act_quant', action='store_true', help='apply activation quantization')
    parser.add_argument('--disable_8bit_head_stem', action='store_true')
    parser.add_argument('--test_before_calibration', action='store_true')

    # weight calibration parameters
    parser.add_argument('--num_samples', default=1024, type=int, help='size of the calibration dataset')
    parser.add_argument('--iters_w', default=20000, type=int, help='number of iteration for adaround')
    parser.add_argument('--weight', default=0.01, type=float, help='weight of rounding cost vs the reconstruction loss.')
    parser.add_argument('--sym', action='store_true', help='symmetric reconstruction, not recommended')
    parser.add_argument('--b_start', default=20, type=int, help='temperature at the beginning of calibration')
    parser.add_argument('--b_end', default=2, type=int, help='temperature at the end of calibration')
    parser.add_argument('--warmup', default=0.2, type=float, help='in the warmup period no regularization is applied')
    parser.add_argument('--step', default=20, type=int, help='record snn output per step')

    # activation calibration parameters
    parser.add_argument('--iters_a', default=5000, type=int, help='number of iteration for LSQ')
    parser.add_argument('--lr', default=4e-4, type=float, help='learning rate for LSQ')
    parser.add_argument('--p', default=2.4, type=float, help='L_p norm minimization for LSQ')

    # GA
    parser.add_argument('--populations', default=10, type=int)
    parser.add_argument('--generations', default=10, type=int)
    parser.add_argument('--num_workers', default=0, type=int, help="number of parallel workers for evaluate")

    args = parser.parse_args()

    run_ga(args)
    #evaluate(args, None)
