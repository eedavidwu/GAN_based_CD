import os
from os import listdir, getcwd

wd=getcwd()

BASE_PATH = wd
PRETRAIN_MODEL_PATH = os.path.join(BASE_PATH,'pretrain')

DATA_PATH = os.path.join(BASE_PATH,'dataset','images')


TRAIN_DATA_PATH = os.path.join(DATA_PATH,'Train')
TRAIN_LABEL_PATH = os.path.join(TRAIN_DATA_PATH,'Ground_truth')
TRAIN_TXT_PATH = os.path.join(BASE_PATH,'dataset','train.txt')

VAL_DATA_PATH = os.path.join(DATA_PATH,'Val')
VAL_LABEL_PATH = os.path.join(VAL_DATA_PATH,'Ground_truth')
VAL_TXT_PATH = os.path.join(BASE_PATH,'dataset','val.txt')

TEST_DATA_PATH = os.path.join(DATA_PATH,'Test')
TEST_LABEL_PATH = os.path.join(VAL_DATA_PATH,'Ground_truth')
TEST_TXT_PATH = os.path.join(BASE_PATH,'dataset','val.txt')

SAVE_PATH = os.path.join(BASE_PATH,'save')
SAVE_CKPT_PATH = os.path.join(SAVE_PATH,'ckpt')

PRETRAIN_VGG16=os.path.join(PRETRAIN_MODEL_PATH,'vgg16.pth')
PRETRAIN_BestCDD=os.path.join(PRETRAIN_MODEL_PATH,'CDD_model_best.pth')

#PRETRAIN_Deeplab=os.path.join(PRETRAIN_MODEL_PATH,'deeplab_v2_voc12.pth')
#TRAINED_BEST_PERFORMANCE_CKPT = os.path.join(SAVE_PATH,'pretrain','CDD_model_best.pth')
TRAINED_BEST_PERFORMANCE_CKPT = os.path.join(SAVE_PATH,'ckpt','model_5.12_10.pth')

##Validate related:
Vali_path = os.path.join(SAVE_PATH,'prediction','contrastive_loss')
save_change_map_dir = os.path.join(Vali_path, 'changemaps/')

epoch = 5
train_batch_size = 8

z_dim = 62
class_num = 2
sample_num = class_num ** 2

INIT_LEARNING_RATE = 1e-4
DECAY = 5e-5
MOMENTUM = 0.90
MAX_ITER = 40000
BATCH_SIZE = 1
THRESHS = [0.1,0.3,0.5]
THRESH = 0.1
LOSS_PARAM_CONV = 3
LOSS_PARAM_FC = 3
TRANSFROM_SCALES= (256,256)
T0_MEAN_VALUE = (87.72, 100.2, 90.43)
T1_MEAN_VALUE = (120.24, 127.93, 121.18)


###GAN realted:
lrG=0.0002
lrD=0.0002
