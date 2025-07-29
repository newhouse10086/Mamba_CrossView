# -*- coding: utf-8 -*-

from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.cuda.amp import autocast,GradScaler
import torch.backends.cudnn as cudnn
import time
from optimizers.make_optimizer_mamba import make_optimizer, print_recommended_config
from models.model import make_model
from datasets.make_dataloader import make_dataset
from tool.utils_server import save_network, save_network_with_name, save_best_model, copyfiles2checkpoints
import warnings
from losses.triplet_loss import Tripletloss,TripletLoss
from losses.cal_loss import cal_kl_loss,cal_loss,cal_triplet_loss
import os

warnings.filterwarnings("ignore")
version =  torch.__version__
#fp16
try:
    from apex.fp16_utils import *
    from apex import amp, optimizers
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
######################################################################
# Options
# --------

def get_parse():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument('--name',default='test', type=str, help='output model name')
    parser.add_argument('--data_dir',default='/home/ma-user/work/Mamba_CrossView/data/University-123/train',type=str, help='training dir path')
    parser.add_argument('--train_all', action='store_true', help='use all training data' )
    parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )
    parser.add_argument('--num_worker', default=6,type=int, help='' )
    parser.add_argument('--batchsize', default=8, type=int, help='batchsize')
    parser.add_argument('--pad', default=0, type=int, help='padding')
    parser.add_argument('--h', default=256, type=int, help='height')
    parser.add_argument('--w', default=256, type=int, help='width')
    parser.add_argument('--views', default=2, type=int, help='the number of views')
    parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
    parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--moving_avg', default=1.0, type=float, help='moving average')
    parser.add_argument('--DA', action='store_true', help='use Color Data Augmentation' )
    parser.add_argument('--share', action='store_true',default=True, help='share weight between different view' )
    parser.add_argument('--fp16', action='store_true',default=False, help='use float16 instead of float32, which will save about 50% memory' )
    parser.add_argument('--autocast', action='store_true',default=True, help='use mix precision' )
    parser.add_argument('--block', default=1, type=int, help='')
    parser.add_argument('--kl_loss', action='store_true',default=False, help='kl_loss' )
    parser.add_argument('--triplet_loss', default=0.3, type=float, help='')
    parser.add_argument('--sample_num', default=1, type=float, help='num of repeat sampling' )
    parser.add_argument('--num_epochs', default=120, type=int, help='' )
    parser.add_argument('--steps', default=[70,110], type=int, help='' )
    parser.add_argument('--backbone', default="VIT-S", type=str, help='VIT-S, MAMBA-S (simplified), MAMBA-V2 (full-featured), MAMBA-LITE (lightweight), VIM-TINY (official), VIM-SMALL (official), VAN-S' )
    parser.add_argument('--pretrain_path', default="", type=str, help='' )
    parser.add_argument('--optimizer', default="auto", type=str, 
                       help='优化器选择: auto(自动), adamw, sgd, sgd_original, lion')
    parser.add_argument('--save_best_only', action='store_true', default=True,
                       help='只保存最佳性能模型 (默认: True)')
    parser.add_argument('--save_checkpoint_freq', default=10, type=int,
                       help='checkpoint保存频率 (每N个epoch保存一次checkpoint，默认: 10)')
    parser.add_argument('--custom_model_name', default="", type=str,
                       help='自定义模型名称前缀 (为空时根据backbone自动选择)')
    opt = parser.parse_args()
    return opt


def train_model(model,opt, optimizer, scheduler, dataloaders,dataset_sizes):
    use_gpu = opt.use_gpu
    num_epochs = opt.num_epochs

    since = time.time()
    warm_up = 0.1  # We start from the 0.1*lrRate
    warm_iteration = round(dataset_sizes['satellite'] / opt.batchsize) * opt.warm_epoch  # first 5 epoch

    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss()
    loss_kl = nn.KLDivLoss(reduction='batchmean')
    triplet_loss = Tripletloss(margin=opt.triplet_loss)
    
    # 初始化最佳性能跟踪
    best_acc = 0.0
    best_epoch = -1
    best_loss = float('inf')
    
    # 根据backbone选择模型名称
    if opt.custom_model_name:
        model_name = opt.custom_model_name
    elif 'VIM-TINY' in opt.backbone:
        model_name = "vim_tiny_patch16_224_FSRA"
    elif 'VIM-SMALL' in opt.backbone:
        model_name = "vim_small_patch16_224_FSRA"
    elif 'MAMBA-LITE' in opt.backbone:
        model_name = "vision_mamba_lite_small_patch16_224_FSRA"
    elif 'MAMBA-V2' in opt.backbone:
        model_name = "vision_mamba_v2_small_patch16_224_FSRA"
    elif 'MAMBA-S' in opt.backbone:
        model_name = "vision_mamba_small_patch16_224_FSRA"
    elif 'VIT-S' in opt.backbone:
        model_name = "vit_small_patch16_224_FSRA"
    elif 'VAN-S' in opt.backbone:
        model_name = "van_small_FSRA"
    else:
        model_name = f"{opt.backbone.lower()}_FSRA"
    
    print(f"\n🎯 最佳模型追踪已启动，模型名称: {model_name}")
    print(f"📁 最佳模型将保存为: {model_name}_best.pth")

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_cls_loss = 0.0
            running_triplet = 0.0
            running_kl_loss = 0.0
            running_loss = 0.0
            running_corrects = 0.0
            running_corrects2 = 0.0
            running_corrects3 = 0.0

            for data,data2,data3 in dataloaders:
                # satallite # street # drone
                loss = 0.0
                # get the inputs
                inputs, labels = data
                inputs2, labels2 = data2
                inputs3, labels3 = data3
                now_batch_size, c, h, w = inputs.shape
                if now_batch_size < opt.batchsize:  # skip the last batch
                    continue
                if use_gpu:
                    inputs = Variable(inputs.cuda().detach())
                    inputs2 = Variable(inputs2.cuda().detach())
                    inputs3 = Variable(inputs3.cuda().detach())
                    labels = Variable(labels.cuda().detach())
                    labels2 = Variable(labels2.cuda().detach())
                    labels3 = Variable(labels3.cuda().detach())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                if phase == 'val':
                    with torch.no_grad():
                        outputs, outputs2 = model(inputs, inputs3)
                else:
                    if opt.views == 2:
                        with autocast():
                            outputs, outputs2 = model(inputs, inputs3) # satellite and drone
                    elif opt.views == 3:
                        outputs, outputs2, outputs3 = model(inputs, inputs2, inputs3)
                f_triplet_loss=torch.tensor((0))
                if opt.triplet_loss>0:
                    features = outputs[1]
                    features2 = outputs2[1]
                    split_num = opt.batchsize//opt.sample_num
                    f_triplet_loss = cal_triplet_loss(features,features2,labels,triplet_loss,split_num)

                    outputs = outputs[0]
                    outputs2 = outputs2[0]

                if isinstance(outputs,list):
                    preds = []
                    preds2 = []
                    for out,out2 in zip(outputs,outputs2):
                        preds.append(torch.max(out.data,1)[1])
                        preds2.append(torch.max(out2.data,1)[1])
                else:
                    _, preds = torch.max(outputs.data, 1)
                    _, preds2 = torch.max(outputs2.data, 1)
                kl_loss = torch.tensor((0))
                if opt.views == 2:
                    cls_loss = cal_loss(outputs, labels,criterion) + cal_loss(outputs2, labels3,criterion) # only compute the global branch
                    #增加klLoss来做mutual learning
                    if opt.kl_loss:
                        kl_loss = cal_kl_loss(outputs,outputs2,loss_kl)

                elif opt.views == 3:
                    if isinstance(outputs,list):
                        preds3 = []
                        for out3 in outputs3:
                            preds3.append(torch.max(out3.data,1)[1])
                        cls_loss = cal_loss(outputs, labels,criterion) + cal_loss(outputs2, labels2,criterion) + cal_loss(outputs3, labels3,criterion)
                        loss+=cls_loss

                    else:
                        _, preds3 = torch.max(outputs3.data, 1)
                        cls_loss = cal_loss(outputs, labels,criterion) + cal_loss(outputs2, labels2,criterion) + cal_loss(outputs3, labels3,criterion)
                        loss+=cls_loss

                loss = kl_loss + cls_loss + f_triplet_loss
                # backward + optimize only if in training phase
                if epoch < opt.warm_epoch and phase == 'train':
                    warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                    loss *= warm_up

                if phase == 'train':
                    if opt.autocast:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()

                # statistics
                if int(version[0]) > 0 or int(version[2]) > 3:  # for the new version like 0.4.0, 0.5.0 and 1.0.0
                    running_loss += loss.item() * now_batch_size
                    running_cls_loss += cls_loss.item()*now_batch_size
                    running_triplet += f_triplet_loss.item() * now_batch_size
                    running_kl_loss += kl_loss.item() * now_batch_size
                else:  # for the old version like 0.3.0 and 0.3.1
                    running_loss += loss.data[0] * now_batch_size
                    running_cls_loss += cls_loss.data[0] * now_batch_size
                    running_triplet += f_triplet_loss.data[0] * now_batch_size
                    running_kl_loss += kl_loss.data[0] * now_batch_size


                if isinstance(preds,list) and isinstance(preds2,list):
                    running_corrects += sum([float(torch.sum(pred == labels.data)) for pred in preds])/len(preds)
                    if opt.views==2:
                        running_corrects2 += sum([float(torch.sum(pred == labels3.data)) for pred in preds2]) / len(preds2)
                    else:
                        running_corrects2 += sum([float(torch.sum(pred == labels2.data)) for pred in preds2])/len(preds2)
                else:
                    running_corrects += float(torch.sum(preds == labels.data))
                    if opt.views == 2:
                        running_corrects2 += float(torch.sum(preds2 == labels3.data))
                    else:
                        running_corrects2 += float(torch.sum(preds2 == labels2.data))
                if opt.views == 3:
                    if isinstance(preds,list) and isinstance(preds2,list):
                        running_corrects3 += sum([float(torch.sum(pred == labels3.data)) for pred in preds3])/len(preds3)
                    else:
                        running_corrects3 += float(torch.sum(preds3 == labels3.data))


            epoch_cls_loss = running_cls_loss/dataset_sizes['satellite']
            epoch_kl_loss = running_kl_loss /dataset_sizes['satellite']
            epoch_triplet_loss = running_triplet/dataset_sizes['satellite']
            epoch_loss = running_loss / dataset_sizes['satellite']
            epoch_acc = running_corrects / dataset_sizes['satellite']
            epoch_acc2 = running_corrects2 / dataset_sizes['satellite']


            lr_backbone = optimizer.state_dict()['param_groups'][0]['lr']
            lr_other = optimizer.state_dict()['param_groups'][1]['lr']
            if opt.views == 2:
                print(
                    '{} Loss: {:.4f} Cls_Loss:{:.4f} KL_Loss:{:.4f} Triplet_Loss {:.4f} Satellite_Acc: {:.4f}  Drone_Acc: {:.4f} lr_backbone:{:.6f} lr_other {:.6f}'
                                                                                .format(phase, epoch_loss,epoch_cls_loss,epoch_kl_loss,
                                                                                        epoch_triplet_loss, epoch_acc,
                                                                                        epoch_acc2,lr_backbone,lr_other))
            elif opt.views == 3:
                epoch_acc3 = running_corrects3 / dataset_sizes['satellite']
                print('{} Loss: {:.4f} Satellite_Acc: {:.4f}  Street_Acc: {:.4f} Drone_Acc: {:.4f}'.format(phase,
                                                                                                           epoch_loss,
                                                                                                           epoch_acc,
                                                                                                           epoch_acc2,
                                                                                                           epoch_acc3))

            # deep copy the model
            if phase == 'train':
                scheduler.step()
                
                # 检查是否是最佳性能
                current_acc = epoch_acc  # 使用当前准确率作为性能指标
                current_loss = epoch_loss
                
                is_best_acc = current_acc > best_acc
                is_best_loss = current_loss < best_loss
                
                # 更新最佳记录
                if is_best_acc:
                    old_best_acc = best_acc  # 保存旧的最佳值用于显示
                    best_acc = current_acc
                    best_epoch = epoch
                    if old_best_acc == 0.0:
                        print(f"🏆 首次设定最佳准确率: {current_acc:.4f}")
                    else:
                        print(f"🏆 发现更好的准确率! {current_acc:.4f} > {old_best_acc:.4f} (之前最佳)")
                    
                    # 保存最佳准确率模型
                    save_best_model(model, opt.name, epoch, current_acc, "accuracy", model_name)
                    print(f"💾 最佳准确率模型已更新并保存")
                
                if is_best_loss:
                    old_best_loss = best_loss
                    best_loss = current_loss  
                    if old_best_loss == float('inf'):
                        print(f"📈 首次设定最佳Loss: {current_loss:.4f}")
                    else:
                        print(f"📈 发现更低的Loss! {current_loss:.4f} < {old_best_loss:.4f} (之前最佳)")
                    
                    # 可选：也可以保存最佳loss模型
                    # save_best_model(model, opt.name, epoch, current_loss, "loss", model_name)
                
                # 每N轮显示当前最佳状态
                if epoch % opt.save_checkpoint_freq == (opt.save_checkpoint_freq - 1):
                    print(f"📊 目前最佳状态:")
                    print(f"   最佳准确率: {best_acc:.4f} (第{best_epoch+1}轮)")
                    print(f"   当前准确率: {current_acc:.4f}")
                    print(f"   最佳Loss: {best_loss:.4f}")
                    print(f"   当前Loss: {current_loss:.4f}")
                    
                    # 每N轮保存一个checkpoint（可选）
                    save_network(model, opt.name, epoch)
                    print(f"📁 第{epoch+1}轮训练checkpoint已保存")
                
                # 显示当前状态
                if is_best_acc:
                    print(f"✨ 第{epoch+1}轮: 准确率 {current_acc:.4f} ⬆️ (新最佳!)")
                else:
                    print(f"📊 第{epoch+1}轮: 准确率 {current_acc:.4f} (最佳: {best_acc:.4f})")
                    
    # 训练结束后的总结
    time_elapsed = time.time() - since
    print(f"\n🎉 训练完成! 耗时: {time_elapsed // 60:.0f}分 {time_elapsed % 60:.0f}秒")
    print(f"   🏆 最佳准确率: {best_acc:.4f} (第{best_epoch+1}轮达到)")
    print(f"   📁 最佳模型已保存为: {model_name}_best_accuracy_{best_acc:.4f}.pth")
    print(f"   📁 最新模型副本: {model_name}_latest.pth")


if __name__ == '__main__':
    opt = get_parse()
    
    # 检查数据路径是否存在
    if not os.path.exists(opt.data_dir):
        print(f"错误：数据路径 {opt.data_dir} 不存在！")
        print("请检查以下几点：")
        print("1. 数据路径是否正确")
        print("2. 数据是否已经下载和解压")
        print("3. 目录结构是否符合要求")
        print("\n预期的数据结构：")
        print(f"{opt.data_dir}/")
        print("├── satellite/")
        print("│   ├── class1/")
        print("│   ├── class2/")
        print("│   └── ...")
        print("├── street/")
        print("│   ├── class1/")
        print("│   ├── class2/")
        print("│   └── ...")
        print("└── drone/")
        print("    ├── class1/")
        print("    ├── class2/")
        print("    └── ...")
        exit(1)
    
    print(f"使用数据路径: {opt.data_dir}")
    print(f"使用backbone: {opt.backbone}")
    
    # 针对不同backbone给出建议
    if opt.backbone == "VIM-TINY":
        print("🎯 使用官方Vision Mamba Tiny（推荐）")
        print("✅ 优势：官方实现，双向状态空间建模，支持预训练权重")
        print("📊 预期性能：78.3% ImageNet Top-1准确率")
        if opt.lr > 0.0005:
            print("💡 建议：Vision Mamba使用较小的学习率，如0.0003-0.0005")
    elif opt.backbone == "VIM-SMALL":
        print("🎯 使用官方Vision Mamba Small（推荐）")
        print("✅ 优势：官方实现，更大模型容量，双向状态空间建模")
        if opt.lr > 0.0005:
            print("💡 建议：Vision Mamba使用较小的学习率，如0.0003-0.0005")
    elif opt.backbone == "MAMBA-S":
        print("⚠️  注意：MAMBA-S是简化版实现，可能收敛困难")
        print("建议：使用VIM-TINY或降低学习率到0.0001")
    elif opt.backbone == "MAMBA-V2":
        print("✅ 使用MAMBA-V2（改进版），具有更好的收敛性")
        if opt.lr > 0.001:
            print("⚠️  建议：Vision Mamba使用更小的学习率，如0.0001")
    elif opt.backbone == "MAMBA-LITE":
        print("🚀 使用MAMBA-LITE（轻量级版本），训练速度更快")
        print("💡 优势：更少参数，更快训练，适合快速实验")
        if opt.lr < 0.001:
            print("💡 提示：MAMBA-LITE可以使用稍大的学习率，如0.001")
    elif opt.backbone == "VIT-S":
        print("✅ 使用ViT-S（稳定可靠的选择）")
    
    print(f"预训练模型路径: {opt.pretrain_path if opt.pretrain_path else '无(随机初始化)'}")
    
    # 模型保存配置信息
    print(f"\n💾 模型保存配置:")
    if opt.custom_model_name:
        model_name = opt.custom_model_name
    elif 'VIM-TINY' in opt.backbone:
        model_name = "vim_tiny_patch16_224_FSRA"
    elif 'VIM-SMALL' in opt.backbone:
        model_name = "vim_small_patch16_224_FSRA"
    elif 'MAMBA-LITE' in opt.backbone:
        model_name = "vision_mamba_lite_small_patch16_224_FSRA"
    elif 'MAMBA-V2' in opt.backbone:
        model_name = "vision_mamba_v2_small_patch16_224_FSRA"
    elif 'MAMBA-S' in opt.backbone:
        model_name = "vision_mamba_small_patch16_224_FSRA"
    elif 'VIT-S' in opt.backbone:
        model_name = "vit_small_patch16_224_FSRA"
    elif 'VAN-S' in opt.backbone:
        model_name = "van_small_FSRA"
    else:
        model_name = f"{opt.backbone.lower()}_FSRA"
    
    print(f"   模型名称: {model_name}")
    print(f"   保存目录: ./checkpoints/{opt.name}/")
    print(f"   保存策略: {'仅保存最佳性能模型' if opt.save_best_only else '每轮保存'}")
    print(f"   最佳模型: {model_name}_best_accuracy_X.XXX.pth")
    print(f"   最新副本: {model_name}_latest.pth")
    print(f"   Checkpoint频率: 每{opt.save_checkpoint_freq}轮保存一次checkpoint")
    print(f"   性能指标: 准确率 (accuracy)")
    
    str_ids = opt.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        gid = int(str_id)
        if gid >= 0:
            gpu_ids.append(gid)

    use_gpu = torch.cuda.is_available()
    opt.use_gpu = use_gpu
    # set gpu ids
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
        cudnn.benchmark = True

    dataloaders,class_names,dataset_sizes = make_dataset(opt)
    opt.nclasses = len(class_names)

    model = make_model(opt)
    
    # 打印推荐的优化器配置
    print_recommended_config(opt.backbone)
    
    # 创建优化器
    print(f"\n🔧 创建优化器 (类型: {opt.optimizer})")
    optimizer_ft, exp_lr_scheduler = make_optimizer(model, opt, optimizer_type=opt.optimizer)

    model = model.cuda()
    #移动文件到指定文件夹
    copyfiles2checkpoints(opt)

    if opt.fp16:
        model, optimizer_ft = amp.initialize(model, optimizer_ft, opt_level="O1")


    train_model(model,opt, optimizer_ft, exp_lr_scheduler,dataloaders,dataset_sizes)
