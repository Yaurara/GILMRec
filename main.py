# coding: utf-8
import os
import argparse

from scipy.sparse.linalg import lgmres

from models import gilmrec
from models.gilmrec import GILMRec
from utils.quick_start import quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='GILMRec', help='name of models') #模型
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of datasets')
    parser.add_argument('--gpu_id', '-g', type=int, default=0, help='gpu number')

    args, _ = parser.parse_known_args()

    config_dict = {
        'gpu_id': args.gpu_id,
    }

    res_1, res_2, res_3, res_4 = quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict,
                                             save_model=True)
    # with open(f"baby_experiment_results.txt", "a") as f:
    #     f.write('\n\n█████████████ BEST ████████████████')
    #     f.write('\tParameters: {}={},\nValid: {},\nTest: {}\n\n'.format(res_1, res_2, res_3, res_4))
