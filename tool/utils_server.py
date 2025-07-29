import os
import torch
import yaml
from models.model import two_view_net, three_view_net
import matplotlib.pyplot as plt
import numpy as np
from shutil import copyfile,copytree,rmtree
import time

def copy_file_or_tree(path,target_dir):
    target_path = os.path.join(target_dir,path)
    if os.path.isdir(path):
        if os.path.exists(target_path):
            rmtree(target_path)
        copytree(path,target_path)
    elif os.path.isfile(path):
         copyfile(path,target_path)

def copyfiles2checkpoints(opt):
    dir_name = os.path.join('./checkpoints', opt.name)
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    # record every run
    copy_file_or_tree('train.py',dir_name)
    copy_file_or_tree('test_server.py',dir_name)
    copy_file_or_tree('evaluate_gpu.py',dir_name)
    copy_file_or_tree('datasets',dir_name)
    copy_file_or_tree('losses',dir_name)
    copy_file_or_tree('models',dir_name)
    copy_file_or_tree('optimizers',dir_name)
    copy_file_or_tree('tool',dir_name)
    copy_file_or_tree('train_test_local.sh',dir_name)
    copy_file_or_tree('heatmap.py',dir_name)

    # save opts
    with open('%s/opts.yaml' % dir_name, 'w') as fp:
        yaml.dump(vars(opt), fp, default_flow_style=False)

def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1 # count the image number in every class
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight

# Get model list for resume
def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        print('no dir: %s'%dirname)
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pth" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name

######################################################################
# Save model
#---------------------------
def save_network(network, dirname, epoch_label):
    if not os.path.isdir('./checkpoints/'+dirname):
        os.mkdir('./checkpoints/'+dirname)
    if isinstance(epoch_label, int):
        save_filename = 'net_%03d.pth'% epoch_label
    else:
        save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join('./checkpoints',dirname,save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available:
        network.cuda()


def save_network_with_name(network, dirname, epoch_label, model_name="vision_mamba_lite_small_patch16_224_FSRA"):
    """
    保存模型，支持自定义模型名称
    
    Args:
        network: 要保存的网络模型
        dirname: 保存目录名
        epoch_label: epoch标签
        model_name: 自定义模型名称
    """
    checkpoint_dir = './checkpoints/' + dirname
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    if isinstance(epoch_label, int):
        save_filename = f'{model_name}_epoch_{epoch_label:03d}.pth'
    else:
        save_filename = f'{model_name}_epoch_{epoch_label}.pth'
    
    save_path = os.path.join(checkpoint_dir, save_filename)
    
    # 保存完整的模型状态
    model_state = {
        'epoch': epoch_label,
        'model_state_dict': network.cpu().state_dict(),
        'model_name': model_name,
        'architecture': getattr(network, '__class__', {}).get('__name__', 'Unknown')
    }
    
    torch.save(model_state, save_path)
    print(f"💾 模型已保存: {save_path}")
    
    # 同时保存一个最新版本的副本
    latest_filename = f'{model_name}_latest.pth'
    latest_path = os.path.join(checkpoint_dir, latest_filename)
    torch.save(model_state, latest_path)
    print(f"💾 最新模型已保存: {latest_path}")
    
    if torch.cuda.is_available():
        network.cuda()
    
    return save_path


def save_best_model(network, dirname, epoch_label, metric_value, metric_name="accuracy", 
                   model_name="vision_mamba_lite_small_patch16_224_FSRA"):
    """
    保存最佳模型
    
    Args:
        network: 要保存的网络模型
        dirname: 保存目录名
        epoch_label: epoch标签
        metric_value: 评估指标值
        metric_name: 评估指标名称
        model_name: 自定义模型名称
    """
    checkpoint_dir = './checkpoints/' + dirname
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    save_filename = f'{model_name}_best_{metric_name}_{metric_value:.4f}.pth'
    save_path = os.path.join(checkpoint_dir, save_filename)
    
    model_state = {
        'epoch': epoch_label,
        'model_state_dict': network.cpu().state_dict(),
        'model_name': model_name,
        'best_metric': {
            'name': metric_name,
            'value': metric_value,
            'epoch': epoch_label
        },
        'architecture': getattr(network, '__class__', {}).get('__name__', 'Unknown')
    }
    
    torch.save(model_state, save_path)
    print(f"🏆 最佳模型已保存: {save_path}")
    
    if torch.cuda.is_available():
        network.cuda()
    
    return save_path


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
        :param tensor: tensor image of size (B,C,H,W) to be un-normalized
        :return: UnNormalized image
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def check_box(images,boxes):
    images = images.permute(0,2,3,1).cpu().detach().numpy()
    boxes = (boxes.cpu().detach().numpy()/16*255).astype(np.int)
    for img,box in zip(images,boxes):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.imshow(img)
        rect = plt.Rectangle(box[0:2], box[2]-box[0], box[3]-box[1])
        ax.add_patch(rect)
        plt.show()



######################################################################
def load_network(opt):
    save_filename = opt.checkpoint

    if opt.views == 2:
        model = two_view_net(opt.nclasses, block=opt.block)
    elif opt.views == 3:
        model = three_view_net(opt.nclasses, opt.droprate, block=opt.block)

    print('Load the model from %s'%save_filename)
    model.load_state_dict(torch.load(save_filename))
    return model

def toogle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)

def update_average(model_tgt, model_src, beta):
    toogle_grad(model_src, False)
    toogle_grad(model_tgt, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert(p_src is not p_tgt)
        p_tgt.copy_(beta*p_tgt + (1. - beta)*p_src)

    toogle_grad(model_src, True)


def load_network_with_name(network, model_path):
    """
    加载使用save_network_with_name保存的模型
    
    Args:
        network: 要加载权重的网络模型
        model_path: 模型文件路径
    
    Returns:
        dict: 包含模型信息的字典
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    print(f"🔄 正在加载模型: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 检查是否是新格式的checkpoint
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # 新格式
        network.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model_info = {
            'epoch': checkpoint.get('epoch', -1),
            'model_name': checkpoint.get('model_name', 'Unknown'),
            'architecture': checkpoint.get('architecture', 'Unknown'),
            'best_metric': checkpoint.get('best_metric', None)
        }
        print(f"✅ 成功加载模型: {model_info['model_name']}")
        print(f"   训练轮数: {model_info['epoch'] + 1}")
        print(f"   模型架构: {model_info['architecture']}")
        if model_info['best_metric']:
            metric = model_info['best_metric']
            print(f"   最佳指标: {metric['name']}={metric['value']:.4f} (epoch {metric['epoch'] + 1})")
    else:
        # 旧格式（直接是state_dict）
        network.load_state_dict(checkpoint, strict=False)
        model_info = {
            'epoch': -1,
            'model_name': 'Legacy Format',
            'architecture': 'Unknown',
            'best_metric': None
        }
        print(f"✅ 成功加载模型 (旧格式)")
    
    if torch.cuda.is_available():
        network.cuda()
    
    return model_info


def list_saved_models(dirname):
    """
    列出指定目录下的所有保存的模型
    
    Args:
        dirname: 模型保存目录名
    
    Returns:
        list: 模型文件列表
    """
    checkpoint_dir = f'./checkpoints/{dirname}'
    if not os.path.exists(checkpoint_dir):
        print(f"❌ 目录不存在: {checkpoint_dir}")
        return []
    
    model_files = []
    for filename in os.listdir(checkpoint_dir):
        if filename.endswith('.pth'):
            filepath = os.path.join(checkpoint_dir, filename)
            file_info = {
                'filename': filename,
                'filepath': filepath,
                'size': os.path.getsize(filepath),
                'modified_time': os.path.getmtime(filepath)
            }
            model_files.append(file_info)
    
    # 按修改时间排序
    model_files.sort(key=lambda x: x['modified_time'], reverse=True)
    
    print(f"📁 在 {checkpoint_dir} 中找到 {len(model_files)} 个模型文件:")
    for i, file_info in enumerate(model_files):
        size_mb = file_info['size'] / (1024 * 1024)
        modified_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(file_info['modified_time']))
        print(f"   {i+1:2d}. {file_info['filename']:<50} ({size_mb:.1f}MB, {modified_time})")
    
    return model_files

