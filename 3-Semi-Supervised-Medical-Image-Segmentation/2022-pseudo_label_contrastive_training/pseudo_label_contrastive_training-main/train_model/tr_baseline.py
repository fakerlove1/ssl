import os

import tensorflow as tf
#config=tf.ConfigProto()
#config.gpu_options.allow_growth=True
#config.allow_soft_placement=True

import matplotlib
matplotlib.use('Agg')

import numpy as np
#to make directories
import pathlib

import sys
sys.path.append('../')

from utils import *
import random
random.seed(1)

import argparse
parser = argparse.ArgumentParser()
#dataset selection
parser.add_argument('--dataset', type=str, default='acdc', choices=['acdc','prostate_md','mmwhs'])
#no of training images
parser.add_argument('--no_of_tr_imgs', type=str, default='tr2', choices=['tr1', 'tr2','tr8','tr10','trall'])
#combination of training images
parser.add_argument('--comb_tr_imgs', type=str, default='c1')
#learning rate of seg unet
parser.add_argument('--lr_seg', type=float, default=0.001)
#version of run
parser.add_argument('--ver', type=int, default=0)

# segmentation loss used for optimization
# 0 for weighted cross entropy, 1 for dice loss w/o background label, 2 for dice loss with background label (default)
parser.add_argument('--dsc_loss', type=int, default=2)

#random deformations - arguments
#enable random deformations
parser.add_argument('--rd_en', type=int, default=1)
#sigma of gaussian distribution used to sample random deformations 3x3 grid values
parser.add_argument('--sigma', type=float, default=5)
#enable random contrasts
parser.add_argument('--ri_en', type=int, default=1)
#enable 1-hot encoding of the labels 
parser.add_argument('--en_1hot', type=int, default=1)
#controls the ratio of deformed images to normal images used in each mini-batch of the training
parser.add_argument('--rd_ni', type=int, default=1)

#no of iterations to run
parser.add_argument('--n_iter', type=int, default=5001)

parse_config = parser.parse_args()
#parse_config = parser.parse_args(args=[])

if parse_config.dataset == 'acdc':
    print('load acdc configs')
    import experiment_init.init_acdc as cfg
    import experiment_init.data_cfg_acdc as data_list
elif parse_config.dataset == 'mmwhs':
    print('load mmwhs configs')
    import experiment_init.init_mmwhs as cfg
    import experiment_init.data_cfg_mmwhs as data_list
elif parse_config.dataset == 'prostate_md':
    print('load prostate_md configs')
    import experiment_init.init_prostate_md as cfg
    import experiment_init.data_cfg_prostate_md as data_list
else:
    raise ValueError(parse_config.dataset)

######################################
# class loaders
# ####################################
#  load dataloader object
from dataloaders import dataloaderObj
dt = dataloaderObj(cfg)

if parse_config.dataset == 'acdc':
    print('set acdc orig img dataloader handle')
    orig_img_dt=dt.load_acdc_imgs
elif parse_config.dataset == 'mmwhs':
    print('set mmwhs orig img dataloader handle')
    orig_img_dt=dt.load_mmwhs_imgs
elif parse_config.dataset == 'prostate_md':
    print('set prostate_md orig img dataloader handle')
    orig_img_dt=dt.load_prostate_imgs_md

# load model object
from models import modelObj
model = modelObj(cfg)
# load f1_utils object
from f1_utils import f1_utilsObj
f1_util = f1_utilsObj(cfg,dt)

if(parse_config.rd_en==1):
    parse_config.en_1hot=1
else:
    parse_config.en_1hot=0
struct_name=cfg.struct_name
val_step_update=cfg.val_step_update
######################################

######################################
# Load training and validation images & labels
######################################
#load training volumes id numbers to train the unet
train_list = data_list.train_data(parse_config.no_of_tr_imgs,parse_config.comb_tr_imgs)
#load saved training data in cropped dimensions directly
print('loading train volumes')
train_imgs, train_labels = dt.load_cropped_img_labels(train_list)
#print('train shape',train_imgs.shape,train_labels.shape)

#load validation volumes id numbers to save the best model during training
val_list = data_list.val_data(parse_config.no_of_tr_imgs,parse_config.comb_tr_imgs)
#load val data both in original dimensions and its cropped dimensions
print('loading val volumes')
val_label_orig,val_img_crop,val_label_crop,pixel_val_list=load_val_imgs(val_list,dt,orig_img_dt)

# get test volumes id list
print('get test volumes list')
test_list = data_list.test_data()
######################################

######################################
#define directory to save the model
save_dir=str(cfg.srt_dir)+'/models/'+str(parse_config.dataset)+'/trained_models/train_baseline/'

save_dir=str(save_dir)+'/with_data_aug/'

if(parse_config.rd_en==1 and parse_config.ri_en==1):
    save_dir=str(save_dir)+'rand_deforms_and_ints_en/'
elif(parse_config.rd_en==1):
    save_dir=str(save_dir)+'rand_deforms_en/'
elif(parse_config.ri_en==1):
    save_dir=str(save_dir)+'rand_ints_en/'

save_dir=str(save_dir)+str(parse_config.no_of_tr_imgs)+'/'+str(parse_config.comb_tr_imgs)+'_v'+str(parse_config.ver)+'/unet_dsc_loss_'+str(parse_config.dsc_loss)+'_n_iter_'+str(parse_config.n_iter)+'_learning_rate_'+str(parse_config.lr_seg)+'/'

print('save dir ',save_dir)
######################################

######################################
# Define network graph
######################################
tf.reset_default_graph()
# Segmentation network
ae = model.seg_unet(learn_rate_seg=parse_config.lr_seg,dsc_loss=parse_config.dsc_loss,\
                     en_1hot=parse_config.en_1hot,mtask_en=0)

# define network/graph to apply random deformations on input images
ae_rd= model.deform_net(batch_size=cfg.mtask_bs)

# define network/graph to apply random contrast and brightness on input images
ae_rc = model.contrast_net(batch_size=cfg.mtask_bs)

# define graph to compute 1-hot encoding of segmentation mask
ae_1hot = model.conv_1hot()
######################################

######################################
# Define checkpoint file to save CNN network architecture and learnt hyperparameters
checkpoint_filename='train_seg_unet_'+str(parse_config.dataset)
logs_path = str(save_dir)+'tensorflow_logs/'
best_model_dir=str(save_dir)+'best_model/'
pathlib.Path(best_model_dir).mkdir(parents=True, exist_ok=True)
######################################

######################################
#writer for train summary
train_writer = tf.summary.FileWriter(logs_path)
#writer for dice score and val summary
#dsc_writer = tf.summary.FileWriter(logs_path)
val_sum_writer = tf.summary.FileWriter(logs_path)
######################################

######################################
# Define session and saver
#sess = tf.Session(config=config)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=2)
######################################

######################################
# parameters values set for training of CNN
mean_f1_val_prev=0.0000001
threshold_f1=0.0000001
n_epochs=parse_config.n_iter
start_epoch=0

tr_loss_list,val_loss_list=[],[]
tr_dsc_list,val_dsc_list=[],[]
ep_no_list=[]
loss_least_val=1
f1_mean_least_val=0.0000000001
######################################

######################################
# loop over all the epochs to train the CNN.
# Randomly sample a batch of images from all data per epoch. On the chosen batch, apply random augmentations and train the network weights.
for epoch_i in range(start_epoch,n_epochs):
    
    # Sample shuffled img, GT labels -- from labeled data
    ld_img_batch,ld_label_batch=shuffle_minibatch_mtask([train_imgs,train_labels],batch_size=cfg.mtask_bs)
    # Apply affine transformations
    ld_img_batch,ld_label_batch=augmentation_function([ld_img_batch,ld_label_batch],dt)
    if(parse_config.rd_en==1 or parse_config.ri_en==1):
        # Apply random augmentations - random deformations + random contrast & brightness values
        ld_img_batch,ld_label_batch=create_rand_augs(cfg,parse_config,sess,ae_rd,ae_rc,ld_img_batch,ld_label_batch)

    #Run optimizer update on the chosen batch of training data from labeled set
    train_summary,loss,_=sess.run([ae['train_summary'],ae['seg_cost'],ae['optimizer_unet_all']],\
                                  feed_dict={ae['x']:ld_img_batch,ae['y_l']:ld_label_batch,ae['train_phase']:True})

    if(epoch_i%val_step_update==0):
        train_writer.add_summary(train_summary, epoch_i)
        train_writer.flush()

    if(epoch_i%cfg.val_step_update==0):
        # Measure validation volumes accuracy in Dice score (DSC) and evaluate validation loss
        # Save the model with the best DSC over validation volumes.
        mean_f1_val_prev,mp_best,mean_total_cost_val,mean_f1=f1_util.track_val_dsc(sess,ae,ae_1hot,saver,mean_f1_val_prev,threshold_f1,\
                                    best_model_dir,val_list,val_img_crop,val_label_crop,val_label_orig,pixel_val_list,\
                                    checkpoint_filename,epoch_i,en_1hot_val=parse_config.en_1hot)

        tr_y_pred=sess.run(ae['y_pred'],feed_dict={ae['x']:ld_img_batch,ae['y_l']:ld_label_batch,ae['train_phase']:False})

        if(parse_config.en_1hot==1):
            tr_accu=f1_util.calc_f1_score(np.argmax(tr_y_pred,axis=-1),np.argmax(ld_label_batch,-1))
        else:
            tr_accu=f1_util.calc_f1_score(np.argmax(tr_y_pred,axis=-1),ld_label_batch)

        tr_dsc_list.append(np.mean(tr_accu))
        val_dsc_list.append(mean_f1)
        tr_loss_list.append(np.mean(loss))
        val_loss_list.append(np.mean(mean_total_cost_val))

        print('epoch_i,loss,f1_val',epoch_i,np.mean(loss),mean_f1_val_prev,mean_f1)

        #Compute and save validation image dice & loss summary
        val_summary_msg = sess.run(ae['val_summary'], feed_dict={ae['mean_dice']: mean_f1, ae['val_totalc']:mean_total_cost_val})
        val_sum_writer.add_summary(val_summary_msg, epoch_i)
        val_sum_writer.flush()

        if(np.mean(mean_f1_val_prev)>f1_mean_least_val):
            f1_mean_least_val=mean_f1_val_prev
            ep_no_list.append(epoch_i)

    if ((epoch_i==n_epochs-1)):
        # model saved at the last epoch of training
        mp = str(save_dir) + str(checkpoint_filename) + '_epochs_' + str(epoch_i) + ".ckpt"
        saver.save(sess, mp)
        try:
            mp_best
        except NameError:
            mp_best=mp
            

######################################
# Plot the training, validation (val) loss and DSC score of training & val images over all the epochs of training.
f1_util.plt_seg_loss([tr_loss_list,val_loss_list],save_dir,title_str='baseline',plt_name='train_seg_loss',ep_no=epoch_i)
f1_util.plt_seg_loss([tr_dsc_list,val_dsc_list],save_dir,title_str='baseline',plt_name='train_dsc_score',ep_no=epoch_i)

sess.close()
######################################

######################################
# find best model checkpoint over all epochs and restore it
mp_best=get_max_chkpt_file(save_dir)
print('mp_best',mp_best)

saver = tf.train.Saver()
#sess = tf.Session(config=config)
sess = tf.Session()
saver.restore(sess, mp_best)
print("Model restored")

######################################
# infer predictions over test volumes from the best model saved during training
struct_name=cfg.struct_name
save_dir_tmp=save_dir+'/test_set_predictions/'
f1_util.test_set_predictions(test_list,sess,ae,dt,orig_img_dt,save_dir_tmp,struct_name)

sess.close()
tf.reset_default_graph()
######################################

