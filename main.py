import numpy as np
import os
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
from torch.nn import functional as F
import utils.transforms as trans
import utils.utils as util
import layer.loss as ls
import utils.metric as mc
import shutil
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch.optim as optim


import cfg.CDD as cfg
import dataset.CDD as data


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

resume = 0

def check_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def untransform(transform_img,mean_vector):

    transform_img = transform_img.transpose(1,2,0)
    transform_img += mean_vector
    transform_img = transform_img.astype(np.uint8)
    transform_img = transform_img[:,:,::-1]
    return transform_img

def various_distance(out_vec_t0, out_vec_t1,dist_flag):
    if dist_flag == 'l2':
        distance = F.pairwise_distance(out_vec_t0, out_vec_t1, p=2)
    if dist_flag == 'l1':
        distance = F.pairwise_distance(out_vec_t0, out_vec_t1, p=1)
    if dist_flag == 'cos':
        distance = 1 - F.cosine_similarity(out_vec_t0, out_vec_t1)
    return distance

def single_layer_similar_heatmap_visual(output_t0,output_t1,save_change_map_dir,epoch,filename,layer_flag,dist_flag):

    interp = nn.Upsample(size=[cfg.TRANSFROM_SCALES[1],cfg.TRANSFROM_SCALES[0]], mode='bilinear')
    n, c, h, w = output_t0.data.shape
    out_t0_rz = torch.transpose(output_t0.view(c, h * w), 1, 0)
    out_t1_rz = torch.transpose(output_t1.view(c, h * w), 1, 0)
    distance = various_distance(out_t0_rz,out_t1_rz,dist_flag=dist_flag)
    similar_distance_map = distance.view(h,w).data.cpu().numpy()
    similar_distance_map_rz = interp(Variable(torch.from_numpy(similar_distance_map[np.newaxis, np.newaxis, :])))
    #similar_dis_map_colorize = cv2.applyColorMap(np.uint8(255 * similar_distance_map_rz.data.cpu().numpy()[0][0]), cv2.COLORMAP_JET)

    similar_distance_map_rz_result=similar_distance_map_rz[0][0].data.cpu().numpy()
    for i in range (similar_distance_map_rz_result.shape[0]):
        for j in range(similar_distance_map_rz_result.shape[1]):
            if similar_distance_map_rz_result[i][j]>0.5:
                similar_distance_map_rz_result[i][j]=255
            else:
                similar_distance_map_rz_result[i][j]=0
    save_change_map_dir_ = os.path.join(save_change_map_dir, 'epoch_' + str(epoch))
    check_dir(save_change_map_dir_)
    save_change_map_dir_layer = os.path.join(save_change_map_dir_,layer_flag)
    check_dir(save_change_map_dir_layer)
    save_weight_fig_dir = os.path.join(save_change_map_dir_layer, filename)
    cv2.imwrite(save_weight_fig_dir, similar_distance_map_rz_result)

    return similar_distance_map_rz.data.cpu().numpy()

def validate(net, val_dataloader,epoch,save_change_map_dir,save_roc_dir):

    net.eval()
    cont_conv5_total,cont_fc_total,cont_embedding_total,num = 0.0,0.0,0.0,0.0
    metric_for_conditions = util.init_metric_for_class_for_cmu(1)
    total_FP = 0.0
    total_FN = 0.0
    total_TP = 0.0
    total_TN = 0.0
    total_F1=0.0
    num=0.0
    for batch_idx, batch in enumerate(val_dataloader):
        inputs1,input2, targets, filename, height, width = batch
        height, width, filename = height.numpy()[0], width.numpy()[0], filename[0]
        inputs1,input2,targets = inputs1.cuda(),input2.cuda(), targets.cuda()
        inputs1,inputs2,targets = Variable(inputs1, volatile=True),Variable(input2,volatile=True) ,Variable(targets)
        out_conv5,out_fc,out_embedding = net(inputs1,inputs2)
        out_conv5_t0, out_conv5_t1 = out_conv5
        out_fc_t0,out_fc_t1 = out_fc
        out_embedding_t0,out_embedding_t1 = out_embedding

        conv5_distance_map = single_layer_similar_heatmap_visual(out_conv5_t0,out_conv5_t1,save_change_map_dir,epoch,filename,'conv5','l2')
        fc_distance_map = single_layer_similar_heatmap_visual(out_fc_t0,out_fc_t1,save_change_map_dir,epoch,filename,'fc','l2')
        embedding_distance_map = single_layer_similar_heatmap_visual(out_embedding_t0,out_embedding_t1,save_change_map_dir,epoch,filename,'embedding','l2')

        prob_change = embedding_distance_map[0][0]

        gt = targets.data.cpu().numpy()
        #FN, FP, posNum, negNum = mc.eval_image_rewrite(gt[0], prob_change, cl_index=1)
        FN, FP, TP, TN = mc.compute_FNFP(gt[0], prob_change)

        Precision =TP / (TP + FP + 1e-10)
        Recall=TP/(TP+FN+1e-10)
        F_score=2*Precision/(Precision+Recall+1e-10)
        OA=(TP+TN)/(TP+TN+FP+FN+1e-10)
        #print('Metirc in Batch:',batch_idx,'in Validation.','Precision:',Precision,'Recall:',Recall,'F1_socre',F_score,'OA:',OA)

        total_FP=total_FP+FP
        total_FN=total_FN+FN
        total_TP=total_TP+TP
        total_TN=total_TN+TN
        num += 1


    pr=total_TP / (total_TP + total_TN + 1e-10)
    recall=total_TP/(total_TP+total_FN+1e-10)
    f_score=2*pr/(pr+recall)
    oa=(total_TP+total_TN)/(total_TP+total_TN+total_TN+total_FN+1e-10)


    print('Metirc in Validation:','Precision:',pr,'Recall:',recall,'F1_socre',f_score,'OA:',oa)
    return f_score,pr,recall,oa



def main():
    ######  load datasets ########
    train_transform_det = trans.Compose([
      trans.Scale(cfg.TRANSFROM_SCALES),
    ])
    val_transform_det = trans.Compose([
      trans.Scale(cfg.TRANSFROM_SCALES),
    ])

    train_data = data.Dataset(cfg.TRAIN_DATA_PATH,cfg.TRAIN_LABEL_PATH,
                                cfg.TRAIN_TXT_PATH,'train',transform=True,
                                transform_med = train_transform_det)
    train_loader = Data.DataLoader(train_data,batch_size=8,
                                 shuffle= True, num_workers= 4, pin_memory= True)
    val_data = data.Dataset(cfg.VAL_DATA_PATH,cfg.VAL_LABEL_PATH,
                            cfg.VAL_TXT_PATH,'val',transform=True,
                            transform_med = val_transform_det)
    val_loader = Data.DataLoader(val_data, batch_size= cfg.BATCH_SIZE,
                                shuffle= False, num_workers= 4, pin_memory= True)

    '''
    test_data = dates.Dataset(cfg.TEST_DATA_PATH,cfg.VAL_LABEL_PATH,
                            cfg.VAL_TXT_PATH,'val',transform=True,
                            transform_med = val_transform_det)
    test_loader = Data.DataLoader(test_data, batch_size= cfg.BATCH_SIZE,
                                shuffle= False, num_workers= 4, pin_memory= True)
    '''

    ######  build  models ########
    import model.DASNET as models
    G = models.generator
    D=models.discriminator

    train_model(D,G, train_loader,val_loader)
    #test_model(model,train_loader)


def train_model(D,G,train_loader,val_loader):
    ###Init data:
    # parameters
    save_dir = cfg.save_dir
    result_dir = cfg.result_dir
    dataset = cfg.dataset
    gpu_mode = cfg.gpu_mode
    model_name = cfg.gan_type
    input_size = cfg.input_size
    z_dim = cfg.z_dim
    class_num = cfg.class_num
    sample_num = cfg.sample_num

    # networks init
    G = G(input_dim=cfg.z_dim, output_dim=data.shape[1], input_size=cfg.input_size,
                       class_num=cfg.class_num)
    D = D(input_dim=data.shape[1], output_dim=1, input_size=self.input_size, class_num=self.class_num)
    G_optimizer = optim.Adam(G.parameters(), lr=cfg.lrG, betas=(args.beta1, args.beta2))
    D_optimizer = optim.Adam(D.parameters(), lr=cfg.lrD, betas=(args.beta1, args.beta2))

    if self.gpu_mode:
        self.G.cuda()
        self.D.cuda()
        self.BCE_loss = nn.BCELoss().cuda()
    else:
        self.BCE_loss = nn.BCELoss()

    print('---------- Networks architecture -------------')
    utils.print_network(self.G)
    utils.print_network(self.D)
    print('-----------------------------------------------')

    # fixed noise & condition
    self.sample_z_ = torch.zeros((self.sample_num, self.z_dim))
    for i in range(self.class_num):
        self.sample_z_[i * self.class_num] = torch.rand(1, self.z_dim)
        for j in range(1, self.class_num):
            self.sample_z_[i * self.class_num + j] = self.sample_z_[i * self.class_num]

    temp = torch.zeros((self.class_num, 1))
    for i in range(self.class_num):
        temp[i, 0] = i

    temp_y = torch.zeros((self.sample_num, 1))
    for i in range(self.class_num):
        temp_y[i * self.class_num: (i + 1) * self.class_num] = temp

    self.sample_y_ = torch.zeros((self.sample_num, self.class_num)).scatter_(1, temp_y.type(torch.LongTensor), 1)
    if self.gpu_mode:
        self.sample_z_, self.sample_y_ = self.sample_z_.cuda(), self.sample_y_.cuda()

    # load dataset
    train_hist = {}
    train_hist['D_loss'] = []
    train_hist['G_loss'] = []
    train_hist['per_epoch_time'] = []
    train_hist['total_time'] = []

    # networks init
    G = model.generator(input_dim=self.z_dim, output_dim=data.shape[1], input_size=self.input_size,
                       class_num=self.class_num)
    D = model.discriminator(input_dim=data.shape[1], output_dim=1, input_size=self.input_size, class_num=self.class_num)
    G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
    D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

    if self.gpu_mode:
        self.G.cuda()
        self.D.cuda()
        self.BCE_loss = nn.BCELoss().cuda()
    else:
        self.BCE_loss = nn.BCELoss()

    print('---------- Networks architecture -------------')
    utils.print_network(self.G)
    utils.print_network(self.D)
    print('-----------------------------------------------')

    # fixed noise & condition
    self.sample_z_ = torch.zeros((self.sample_num, self.z_dim))
    for i in range(self.class_num):
        self.sample_z_[i * self.class_num] = torch.rand(1, self.z_dim)
        for j in range(1, self.class_num):
            self.sample_z_[i * self.class_num + j] = self.sample_z_[i * self.class_num]

    temp = torch.zeros((self.class_num, 1))
    for i in range(self.class_num):
        temp[i, 0] = i

    temp_y = torch.zeros((self.sample_num, 1))
    for i in range(self.class_num):
        temp_y[i * self.class_num: (i + 1) * self.class_num] = temp

    self.sample_y_ = torch.zeros((self.sample_num, self.class_num)).scatter_(1, temp_y.type(torch.LongTensor), 1)
    if self.gpu_mode:
        self.sample_z_, self.sample_y_ = self.sample_z_.cuda(), self.sample_y_.cuda()



    best_metric = 0
    save_dir = cfg.SAVE_CKPT_PATH
    #pretrain_deeplab_path = cfg.PRETRAIN_VGG16
    pretrain_path = cfg.PRETRAIN_BestCDD

    if resume:
        checkpoint = torch.load(cfg.TRAINED_BEST_PERFORMANCE_CKPT)
        model.load_state_dict(checkpoint['state_dict'])
        print('resume success')
        print('Load the model:',cfg.TRAINED_BEST_PERFORMANCE_CKPT)
    else:
        pretrain_model = torch.load(pretrain_path)
        model.load_state_dict(pretrain_model['state_dict'],strict=False)
        #model.init_parameters_from_deeplab(pretrain_model)
        print('Load the model:',pretrain_model)

    model = model.cuda()
    MaskLoss = ls.DContrastiveLoss()

    #########
    ######### optimizer ##########
    ######## how to set different learning rate for differernt layers #########
    optimizer = torch.optim.Adam(params=model.parameters(),lr=cfg.INIT_LEARNING_RATE,weight_decay=cfg.DECAY)
    ######## iter img_label pairs ###########
    loss_total = 0
    for epoch in range(100):
        for batch_idx, batch in enumerate(train_loader):
            step = epoch * len(train_loader) + batch_idx
            util.adjust_learning_rate(cfg.INIT_LEARNING_RATE, optimizer, step)
            model.train()
            img1_idx,img2_idx,label_idx, filename,height,width = batch
            img1,img2,label = Variable(img1_idx.cuda()),Variable(img2_idx.cuda()),Variable(label_idx.cuda())
            label = label.float()/255 #1: changed
            out_conv5, out_fc,out_embedding = model(img1, img2)
            out_conv5_t0,out_conv5_t1 = out_conv5
            out_fc_t0,out_fc_t1 = out_fc
            out_embedding_t0,out_embedding_t1 = out_embedding
            label_rz_conv5 = util.rz_label(label,size=out_conv5_t0.data.cpu().numpy().shape[2:]).cuda()
            label_rz_fc = util.rz_label(label,size=out_fc_t0.data.cpu().numpy().shape[2:]).cuda()
            label_rz_embedding = util.rz_label(label,size=out_embedding_t0.data.cpu().numpy().shape[2:]).cuda()
            contractive_loss_conv5 = MaskLoss(out_conv5_t0,out_conv5_t1,label_rz_conv5)
            contractive_loss_fc = MaskLoss(out_fc_t0,out_fc_t1,label_rz_fc)
            contractive_loss_embedding = MaskLoss(out_embedding_t0,out_embedding_t1,label_rz_embedding)
            loss = contractive_loss_conv5 + contractive_loss_fc + contractive_loss_embedding
            #loss=contractive_loss_conv5
            loss_total += loss.data.cpu()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (batch_idx) % 50 == 0:
                print("Epoch [%d/%d] Loss: %.4f Mask_Loss_conv5: %.4f Mask_Loss_fc: %.4f "
                  "Mask_Loss_embedding: %.4f" % (epoch, batch_idx,loss.data[0],contractive_loss_conv5.data[0],
                                                 contractive_loss_fc.data[0],contractive_loss_embedding.data[0]))

            if (batch_idx) % 500 == 0 and batch_idx !=0:
                model.eval()
                f_score, pr, recall, oa= validate(model, val_loader, epoch,cfg.save_change_map_dir, cfg.save_roc_dir)
                current_metric=f_score
                if current_metric > best_metric:
                    torch.save({'state_dict': model.state_dict()},os.path.join(save_dir, 'model' + str(epoch) + '.pth'))
                    shutil.copy(os.path.join(save_dir, 'model' + str(epoch) + '.pth'),os.path.join(save_dir, 'model_best.pth'))
                    print('save best model with F_socore=',f_score,' Pr=',pr,' Recall=',recall,' OA=',oa,'in Epoch:',epoch)
                    best_metric = current_metric

        f_score, pr, recall, oa = validate(model, val_loader, epoch, cfg.save_change_map_dir,cfg.save_roc_dir)
        current_metric = f_score
        if current_metric > best_metric:
            torch.save({'state_dict': model.state_dict()},
                     os.path.join(save_dir, 'model' + str(epoch) + '.pth'))
            shutil.copy(os.path.join(save_dir, 'model' + str(epoch) + '.pth'),
                     os.path.join(save_dir, 'model_best.pth'))
            best_metric = current_metric
            print('save new model with F_socore=', f_score, ' Pr=', pr, ' Recall=', recall, ' OA=', oa,'in Epoch:',epoch)

        if epoch % 2 == 0:
            torch.save({'state_dict': model.state_dict()},
                   os.path.join(cfg.SAVE_CKPT_PATH, 'model_Best_CDD_5.12_' + str(epoch) + '.pth'))

def test_model(model,test_loader):

    Load_weight_path='/data/wuhaotian/CDD/Dataset/save/ckpt/model_5.8_70.pth'
    checkpoint = torch.load(Load_weight_path)
    model.load_state_dict(checkpoint['state_dict'])
    print('Load model success')
    model=model.cuda()

    save_change_map_dir = cfg.save_change_map_dir
    save_valid_dir = cfg.save_valid_dir
    save_roc_dir = cfg.save_roc_dir

    for batch_idx, batch in enumerate(test_loader):
        model.eval()
        img1_idx, img2_idx, label_idx, filename, height, width = batch
        img1, img2, label = Variable(img1_idx.cuda()), Variable(img2_idx.cuda()), Variable(label_idx.cuda())
        filename=filename[0]
        out_conv5, out_fc, out_embedding = model(img1, img2)
        out_conv5_t0, out_conv5_t1 = out_conv5
        out_fc_t0, out_fc_t1 = out_fc
        out_embedding_t0, out_embedding_t1 = out_embedding

        conv5_distance_map = single_layer_similar_heatmap_visual(out_conv5_t0, out_conv5_t1, save_change_map_dir, 'test',
                                                                 filename, 'conv5', 'l2')
        fc_distance_map = single_layer_similar_heatmap_visual(out_fc_t0, out_fc_t1, save_change_map_dir, 'test', filename,
                                                              'fc', 'l2')
        embedding_distance_map = single_layer_similar_heatmap_visual(out_embedding_t0, out_embedding_t1,
                                                                     save_change_map_dir, 'test', filename, 'embedding',
                                                                     'l2')
        print('finish test:',filename)
        GT_Path='/data/wuhaotian/CDD/Dataset/Train/Ground_truth/'+filename
        GT_img = Image.open(GT_Path)
        plt.imshow(GT_img)
        plt.show()

        plt.imshow(conv5_distance_map[0][0])
        plt.show()
        print('finish')


if __name__ == '__main__':
   main()
