
from vaemodel import Model
import numpy as np
import pickle
import torch
import os
import argparse
import random
import  time
import sys

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def GetNowTime():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))

def getTimestamp():
    return time.strftime("%m%d%H%M%S", time.localtime(time.time()))



parser = argparse.ArgumentParser()

parser.add_argument('--dataset',default='AWA2')
parser.add_argument('--num_shots',type=int,default=0)
parser.add_argument('--generalized', type = str2bool,default=True)
parser.add_argument('--max_weight', type = float, default=20)
parser.add_argument('--min_weight', type = float, default=0.00001)
parser.add_argument('--entr_weight', type = float, default=9)
parser.add_argument('--da_weight', type = float, default=1)
parser.add_argument('--seed', type = int, default=42)
parser.add_argument('--epoch', type = int, default=120)
parser.add_argument('--cls_batch_size', type = int, default=32)
parser.add_argument('--latent_size', type = int, default=76)
parser.add_argument('--mi_begin_epoch', type = int, default=0)
parser.add_argument('--save_mode', type= str, default='run', help='save|load|run')
parser.add_argument('--load_ts', type= str, default='model_name') 
parser.add_argument('--ratio_lv', type= int, default=0) 
parser.add_argument('--res_1', type= int, default=1560)
parser.add_argument('--res_2', type= int, default=1660) 
parser.add_argument('--att_1', type= int, default=1450) 
parser.add_argument('--att_2', type= int, default=665) 
parser.add_argument('--beta', type= float, default=0.25) 
parser.add_argument('--ca', type= float, default=2.37) 
parser.add_argument('--da', type= float, default=8.13) 
parser.add_argument('--gen_seen', type= int, default=200) 
parser.add_argument('--gen_novel', type= int, default=750) 
parser.add_argument('--zsL_gen_novel', type= int, default=200) 
args = parser.parse_args()

########################################
# the basic hyperparameters
########################################
hyperparameters = {
    'num_shots': 0,
    'device': 'cuda',
    'model_specifics': {'cross_reconstruction': True,
                       'name': 'CADA',
                       'distance': 'wasserstein',
                       'warmup': {'beta': {'factor': args.beta,
                                           'end_epoch': 93,
                                           'start_epoch': 0},
                                  'cross_reconstruction': {'factor': args.ca,
                                                           'end_epoch': 75,
                                                           'start_epoch': 21},
                                  'distance': {'factor': args.da,
                                               'end_epoch': 22,
                                               'start_epoch': 6}}},

    'lr_gen_model': 0.00015,
    'generalized': True,
    'batch_size': 50,
    'xyu_samples_per_class': {'SUN': (200, 0, 400, 0),
                              'APY': (200, 0, 400, 0),
                              'CUB': (200, 0, 400, 0),
                              'AWA2': (200, 0, 400, 0),
                              'FLO': (200, 0, 400, 0),
                              'AWA1': (200, 0, 400, 0)},
    'epochs': args.epoch,
    'loss': 'l1',
    'auxiliary_data_source' : 'attributes',
    'lr_cls': 0.001,
    'dataset': 'CUB',
    'hidden_size_rule': {'resnet_features': (args.res_1, args.res_2),
                        'attributes': (args.att_1, args.att_2),
                        'sentences': (1450, 665) },
    'latent_size': args.latent_size
}

# The training epochs for the final classifier, for early stopping,
# as determined on the validation spit

cls_train_steps = [
      {'dataset': 'SUN',  'num_shots': 0, 'generalized': True, 'cls_train_steps': 21},
      {'dataset': 'SUN',  'num_shots': 0, 'generalized': False, 'cls_train_steps': 30},
      {'dataset': 'SUN',  'num_shots': 1, 'generalized': True, 'cls_train_steps': 22},
      {'dataset': 'SUN',  'num_shots': 1, 'generalized': False, 'cls_train_steps': 96},
      {'dataset': 'SUN',  'num_shots': 5, 'generalized': True, 'cls_train_steps': 29},
      {'dataset': 'SUN',  'num_shots': 5, 'generalized': False, 'cls_train_steps': 78},
      {'dataset': 'SUN',  'num_shots': 2, 'generalized': True, 'cls_train_steps': 29},
      {'dataset': 'SUN',  'num_shots': 2, 'generalized': False, 'cls_train_steps': 61},
      {'dataset': 'SUN',  'num_shots': 10, 'generalized': True, 'cls_train_steps': 79},
      {'dataset': 'SUN',  'num_shots': 10, 'generalized': False, 'cls_train_steps': 94},
      {'dataset': 'AWA1', 'num_shots': 0, 'generalized': True, 'cls_train_steps': 33},
      {'dataset': 'AWA1', 'num_shots': 0, 'generalized': False, 'cls_train_steps': 25},
      {'dataset': 'AWA1', 'num_shots': 1, 'generalized': True, 'cls_train_steps': 40},
      {'dataset': 'AWA1', 'num_shots': 1, 'generalized': False, 'cls_train_steps': 81},
      {'dataset': 'AWA1', 'num_shots': 5, 'generalized': True, 'cls_train_steps': 89},
      {'dataset': 'AWA1', 'num_shots': 5, 'generalized': False, 'cls_train_steps': 62},
      {'dataset': 'AWA1', 'num_shots': 2, 'generalized': True, 'cls_train_steps': 56},
      {'dataset': 'AWA1', 'num_shots': 2, 'generalized': False, 'cls_train_steps': 59},
      {'dataset': 'AWA1', 'num_shots': 10, 'generalized': True, 'cls_train_steps': 100},
      {'dataset': 'AWA1', 'num_shots': 10, 'generalized': False, 'cls_train_steps': 50},
      {'dataset': 'CUB',  'num_shots': 0, 'generalized': True, 'cls_train_steps': 23},
      {'dataset': 'CUB',  'num_shots': 0, 'generalized': False, 'cls_train_steps': 22},
      {'dataset': 'CUB',  'num_shots': 1, 'generalized': True, 'cls_train_steps': 34},
      {'dataset': 'CUB',  'num_shots': 1, 'generalized': False, 'cls_train_steps': 46},
      {'dataset': 'CUB',  'num_shots': 5, 'generalized': True, 'cls_train_steps': 64},
      {'dataset': 'CUB',  'num_shots': 5, 'generalized': False, 'cls_train_steps': 73},
      {'dataset': 'CUB',  'num_shots': 2, 'generalized': True, 'cls_train_steps': 39},
      {'dataset': 'CUB',  'num_shots': 2, 'generalized': False, 'cls_train_steps': 31},
      {'dataset': 'CUB',  'num_shots': 10, 'generalized': True, 'cls_train_steps': 85},
      {'dataset': 'CUB',  'num_shots': 10, 'generalized': False, 'cls_train_steps': 67},
      {'dataset': 'AWA2', 'num_shots': 0, 'generalized': True, 'cls_train_steps': 29},
      {'dataset': 'AWA2', 'num_shots': 0, 'generalized': False, 'cls_train_steps': 39},
      {'dataset': 'AWA2', 'num_shots': 1, 'generalized': True, 'cls_train_steps': 44},
      {'dataset': 'AWA2', 'num_shots': 1, 'generalized': False, 'cls_train_steps': 96},
      {'dataset': 'AWA2', 'num_shots': 5, 'generalized': True, 'cls_train_steps': 99},
      {'dataset': 'AWA2', 'num_shots': 5, 'generalized': False, 'cls_train_steps': 100},
      {'dataset': 'AWA2', 'num_shots': 2, 'generalized': True, 'cls_train_steps': 69},
      {'dataset': 'AWA2', 'num_shots': 2, 'generalized': False, 'cls_train_steps': 79},
      {'dataset': 'AWA2', 'num_shots': 10, 'generalized': True, 'cls_train_steps': 86},
      {'dataset': 'AWA2', 'num_shots': 10, 'generalized': False, 'cls_train_steps': 78},
      {'dataset': 'APY', 'num_shots': 0, 'generalized': False, 'cls_train_steps': 78},
      {'dataset': 'APY', 'num_shots': 0, 'generalized': True, 'cls_train_steps': 78},
      {'dataset': 'FLO', 'num_shots': 0, 'generalized': False, 'cls_train_steps': 78},
      {'dataset': 'FLO', 'num_shots': 0, 'generalized': True, 'cls_train_steps': 78}
      ]

##################################
# change some hyperparameters here
##################################
hyperparameters['dataset'] = args.dataset
hyperparameters['num_shots']= args.num_shots
hyperparameters['generalized']= args.generalized

hyperparameters['cls_train_steps'] = [x['cls_train_steps']  for x in cls_train_steps
                                        if all([hyperparameters['dataset']==x['dataset'],
                                        hyperparameters['num_shots']==x['num_shots'],
                                        hyperparameters['generalized']==x['generalized'] ])][0]

print('***')
print(hyperparameters['cls_train_steps'] )
if hyperparameters['generalized']:
    if hyperparameters['num_shots']==0:
        hyperparameters['samples_per_class'] = {'CUB': (args.gen_seen, 0, args.gen_novel, 0), 'SUN': (args.gen_seen, 0, args.gen_novel, 0),
                                'APY': (args.gen_seen, 0,  args.gen_novel, 0), 'AWA1': (args.gen_seen, 0, args.gen_novel, 0),
                                'AWA2': (args.gen_seen, 0, args.gen_novel, 0), 'FLO': (args.gen_seen, 0, args.gen_novel, 0)}
else:
    if hyperparameters['num_shots']==0:
        hyperparameters['samples_per_class'] = {'CUB': (0, 0, args.zsL_gen_novel, 0), 'SUN': (0, 0, args.zsL_gen_novel, 0),
                                                    'APY': (0, 0, args.zsL_gen_novel, 0), 'AWA1': (0, 0, args.zsL_gen_novel, 0),
                                                    'AWA2': (0, 0, args.zsL_gen_novel, 0), 'FLO': (0, 0, args.zsL_gen_novel, 0)}

def fix_seed(seed):
    print("Random Seed: ", seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False




fix_seed(args.seed)

print(args)
print(GetNowTime())
print('Begin run!!!')
since = time.time()




model = Model( hyperparameters, args)
model.to(hyperparameters['device'])


if args.save_mode == 'save' or args.save_mode == 'run':
    print('Beging trainging models......')
    losses = model.my_train_vae()
    if losses == 'nan':
        print('Loss is NaN!!!')
        exit(0)
    torch.cuda.empty_cache()

if args.save_mode == 'save':
    time_post = getTimestamp()
    saveStr = 'Data_'+args.dataset+'_G_'+str(args.generalized)+'_max_'+str(args.max_weight)+'_min_'+str(args.min_weight)+'_entr_'+str(args.entr_weight)+'_latent_'+str(args.latent_size)+'_E_'+str(args.epoch)+'_Seed_'+str(args.seed)+'_T_'+time_post
    print('Saving model: {:s}'.format(saveStr))
    state = {
        'v2z_encoder': model.encoder['resnet_features'].state_dict(),
        's2z_encoder': model.encoder['attributes'].state_dict(),
        'z2v_decoder': model.decoder['resnet_features'].state_dict(),
        'z2s_decoder': model.decoder['attributes'].state_dict(),
        'time_post':time_post
    }
    torch.save(state, './saved_model/' + saveStr)
    print('Saved!')


if args.save_mode == 'load':
    print('Loading model time_post: {:s}'.format(args.load_ts))
    state = torch.load('./saved_model/'+args.load_ts)
    model.encoder['resnet_features'].load_state_dict(state['v2z_encoder'])
    model.encoder['attributes'].load_state_dict(state['s2z_encoder'])
    model.decoder['resnet_features'].load_state_dict(state['z2v_decoder'])
    model.decoder['attributes'].load_state_dict(state['z2s_decoder'])

model.mytrain_classifier()

time_elapsed = time.time() - since
print('End run!!!')
print('Time Elapsed: {}'.format(time_elapsed))
print(GetNowTime())
sys.stdout.flush()
