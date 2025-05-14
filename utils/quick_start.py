# coding: utf-8

from logging import getLogger
from itertools import product
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.logger import init_logger
from utils.configurator import Config
from utils.utils import init_seed, get_model, get_trainer, dict2str
import platform
import os


def quick_start(model, dataset, config_dict, save_model=True):
    # merge config dict
    config = Config(model, dataset, config_dict)
    init_logger(config)
    logger = getLogger()
    # print config infor
    logger.info('██Server: \t' + platform.node())
    logger.info('██Dir: \t' + os.getcwd() + '\n')
    logger.info(config)

    # load data
    dataset = RecDataset(config)
    # print dataset statistics
    logger.info(str(dataset))

    #划分数据集
    train_dataset, valid_dataset, test_dataset = dataset.split()
    logger.info('\n====Training====\n' + str(train_dataset))
    logger.info('\n====Validation====\n' + str(valid_dataset))
    logger.info('\n====Testing====\n' + str(test_dataset))

    # wrap into dataloader
    train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    (valid_data, test_data) = (
        EvalDataLoader(config, valid_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']),
        EvalDataLoader(config, test_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']))




    ############ Dataset loadded, run model
    hyper_ret = [] #用于存储超参数相关的返回结果
    val_metric = config['valid_metric'].lower() #获取配置中指定的验证指标（如准确率、召回率等），并将其转换为小写
    best_test_value = 0.0
    idx = best_test_idx = 0 #最佳模型的索引

    logger.info('\n\n=================================\n\n')


    # hyper-parameters
    hyper_ls = [] #用于存储超参数的取值列表
    if "seed" not in config['hyper_parameters']:
        config['hyper_parameters'] = ['seed'] + config['hyper_parameters']
    for i in config['hyper_parameters']: #遍历列表中的每个超参数
        hyper_ls.append(config[i] or [None]) #从配置字典 config 中获取它的值。如果该值为 None，则使用 [None] 作为默认值。每个超参数可能有多个取值，或None
    # 计算超参数组合的总数
    combinators = list(product(*hyper_ls)) #所有组合
    total_loops = len(combinators) #计算所有超参数组合的总数
    #遍历所有超参数组合并训练模型
    for hyper_tuple in combinators:

        # 设置每个超参数组合的参数
        for j, k in zip(config['hyper_parameters'], hyper_tuple):
            config[j] = k
        # random seed reset
        init_seed(config['seed'])

        logger.info('========={}/{}: Parameters:{}={}======='.format(
            idx+1, total_loops, config['hyper_parameters'], hyper_tuple))

        #设置训练数据加载器
        train_data.pretrain_setup()
        #加载并初始化模型
        model = get_model(config['model'])(config, train_data).to(config['device'])
        logger.info(model)

        #加载并初始化训练器
        trainer = get_trainer()(config, model)
        # debug
        #训练模型并评估
        best_valid_score, best_valid_result, best_test_upon_valid = trainer.fit(train_data, valid_data=valid_data, test_data=test_data, saved=save_model)
        #########
        hyper_ret.append((hyper_tuple, best_valid_result, best_test_upon_valid))

        #更新最佳测试结果
        if best_test_upon_valid[val_metric] > best_test_value:
            best_test_value = best_test_upon_valid[val_metric]
            best_test_idx = idx
        idx += 1

        logger.info('best valid result: {}'.format(dict2str(best_valid_result)))
        logger.info('test result: {}'.format(dict2str(best_test_upon_valid)))
        logger.info('████Current BEST████:\nParameters: {}={},\n'
                    'Valid: {},\nTest: {}\n\n\n'.format(config['hyper_parameters'],
            hyper_ret[best_test_idx][0], dict2str(hyper_ret[best_test_idx][1]), dict2str(hyper_ret[best_test_idx][2])))

    # 输出所有超参数组合的结果
    logger.info('\n============All Over=====================')
    for (p, k, v) in hyper_ret:
        logger.info('Parameters: {}={},\n best valid: {},\n best test: {}'.format(config['hyper_parameters'],
                                                                                  p, dict2str(k), dict2str(v)))
#输出最终的最佳结果
    logger.info('\n\n█████████████ BEST ████████████████')
    logger.info('\tParameters: {}={},\nValid: {},\nTest: {}\n\n'.format(config['hyper_parameters'],
                                                                   hyper_ret[best_test_idx][0],
                                                                   dict2str(hyper_ret[best_test_idx][1]),
                                                                   dict2str(hyper_ret[best_test_idx][2])))


    return config['hyper_parameters'] ,  hyper_ret[best_test_idx][0] ,dict2str(hyper_ret[best_test_idx][1]),dict2str(hyper_ret[best_test_idx][2])
    # import numpy as np
    # _, i_v_feats = model.agg_mm_neighbors('v')
    # np.save('log/'+config['model'] + '-' + config['dataset']+'-i_v_feats', i_v_feats.detach().cpu().numpy())
    # _, i_t_feats = model.agg_mm_neighbors('t')
    # np.save('log/'+config['model'] + '-' + config['dataset']+'-i_t_feats', i_t_feats.detach().cpu().numpy())

