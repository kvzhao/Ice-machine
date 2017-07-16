import os
import argparse
import importlib

from improved_wgan import WassersteinGAN
from sampler import DataSampler, NoiseSampler


def main():
    parser = argparse.ArgumentParser('')
    parser.add_argument('--gpus', type=str, default='0')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    model = importlib.import_module('models')
    data  = importlib.import_module('sampler')
    
    xs = DataSampler()
    zs = NoiseSampler()

    G = model.Generator()
    D = model.Discriminator()

    wgan = WassersteinGAN(G, D, xs, zs)
    wgan.train()


if __name__ == '__main__':
    main()