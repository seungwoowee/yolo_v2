import argparse
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from data.logo_dataset import LOGODataset

from src.util.utils import *
from src.util.loss import YoloLoss
from src.models.yolo_net import Yolo
from tensorboardX import SummaryWriter
import shutil

import logging


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=448,
                        help="The common width and height for all images")  # default 448
    parser.add_argument("--batch_size", type=int, default=5, help="The number of images per batch")  # default 10
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--decay", type=float, default=0.0005)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num_epoches", type=int, default=160)
    parser.add_argument("--test_interval", type=int, default=5, help="Number of epoches between testing phases")
    parser.add_argument("--object_scale", type=float, default=1.0)
    parser.add_argument("--noobject_scale", type=float, default=0.5)
    parser.add_argument("--class_scale", type=float, default=1.0)
    parser.add_argument("--coord_scale", type=float, default=5.0)
    parser.add_argument("--reduction", type=int, default=32)
    parser.add_argument("--es_min_delta", type=float, default=0.0,
                        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    parser.add_argument("--es_patience", type=int, default=0,
                        help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")
    parser.add_argument("--train_set", type=str, default="train")
    parser.add_argument("--val_set", type=str, default="val")
    parser.add_argument("--img_path", type=str, default="../../dataset/YOLO/input_images_with_LOGO/images",
                        help="the root folder of dataset")
    parser.add_argument("--anno_path", type=str, default="../../dataset/YOLO/input_images_with_LOGO/anno_pickle",
                        help="the root folder of dataset")
    parser.add_argument("--tensorboard_path", type=str, default="../tensorboard")
    parser.add_argument("--experiments_root", type=str, default="../experiments")
    parser.add_argument("--model_name", type=str, default="yolo_v2_448")

    parser.add_argument("--resume", type=str, default=None)
    # parser.add_argument("--resume", type=str, default=True)
    parser.add_argument("--pre_trained_model_path", type=str, default="trained_models/yolo_v2_30.pth")
    parser.add_argument("--pre_trained_model_state", type=str, default="trained_models/yolo_v2_30.state")

    args = parser.parse_args()
    return args


def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    learning_rate_schedule = {"0": 1e-5, "5": 1e-4,
                              "80": 1e-5, "110": 1e-6}
    training_params = {"batch_size": opt.batch_size,
                       "shuffle": True,
                       "drop_last": True,
                       "collate_fn": custom_collate_fn}

    val_params = {"batch_size": opt.batch_size,
                  "shuffle": False,
                  "drop_last": False,
                  "collate_fn": custom_collate_fn}

    training_set = LOGODataset(opt.img_path, opt.anno_path, opt.train_set, opt.image_size)
    training_generator = DataLoader(training_set, **training_params)

    val_set = LOGODataset(opt.img_path, opt.anno_path, opt.val_set, opt.image_size, is_training=False)
    test_generator = DataLoader(val_set, **val_params)

    if torch.cuda.is_available():
        if opt.resume is None:
            mkdir_and_rename(os.path.join(opt.experiments_root, opt.model_name))
            # model = torch.load(opt.pre_trained_model_path, map_location=lambda storage, loc: storage)
            model = Yolo(training_set.num_classes)
            start_epoch = 0

        else:
            model = Yolo(training_set.num_classes)
            load_net = torch.load(opt.pre_trained_model_path)

            load_net_clean = OrderedDict()
            for k, v in load_net.items():
                if k.startswith('module.'):
                    load_net_clean[k[7:]] = v
                else:
                    load_net_clean[k] = v
            model.load_state_dict(load_net_clean, strict=True)
            resume_state = torch.load(opt.pre_trained_model_state)
            start_epoch = resume_state['epoch']


    else:
        if opt.resume is None:
            mkdir_and_rename(os.path.join(opt.experiments_root, opt.model_name))
            # model = torch.load(opt.pre_trained_model_path, map_location=lambda storage, loc: storage)
            model = Yolo(training_set.num_classes)
            start_epoch = 0

        else:
            pre_trained_model_path = os.path.join(opt.experiments_root, opt.model_name, opt.pre_trained_model_path)
            model = Yolo(training_set.num_classes)
            load_net = torch.load(pre_trained_model_path)

            load_net_clean = OrderedDict()
            for k, v in load_net.items():
                if k.startswith('module.'):
                    load_net_clean[k[7:]] = v
                else:
                    load_net_clean[k] = v
            model.load_state_dict(load_net_clean, strict=True)

            pre_trained_state_path = os.path.join(opt.experiments_root, opt.model_name, opt.pre_trained_model_state)
            resume_state = torch.load(pre_trained_state_path)
            start_epoch = resume_state['epoch']

    # The following line will re-initialize weight for the last layer, which is useful
    # when you want to retrain the model based on my trained weights. if you uncomment it,
    # you will see the loss is already very small at the beginning.
    # nn.init.normal_(list(model.modules())[-1].weight, 0, 0.01)
    # tensorboard_path = os.path.join(opt.tensorboard_path, "{}".format(opt.channel))

    tensorboard_path = os.path.join(opt.tensorboard_path, opt.model_name)

    if os.path.isdir(tensorboard_path):
        shutil.rmtree(tensorboard_path)

    os.makedirs(tensorboard_path)
    tb_logger = SummaryWriter(log_dir=tensorboard_path)

    if torch.cuda.is_available():
        tb_logger.add_graph(model.cpu(), torch.rand(opt.batch_size, 3, opt.image_size, opt.image_size))
        model.cuda()
    else:
        tb_logger.add_graph(model, torch.rand(opt.batch_size, 3, opt.image_size, opt.image_size))

    setup_logger('base', os.path.join(opt.experiments_root, opt.model_name, 'train'), level=logging.INFO, screen=True,
                 tofile=True)

    setup_logger('val', os.path.join(opt.experiments_root, opt.model_name, 'val'), level=logging.INFO, screen=True,
                 tofile=True)
    logger = logging.getLogger('base')
    logger.info(opt)

    criterion = YoloLoss(training_set.num_classes, model.anchors, opt.reduction)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=opt.momentum, weight_decay=opt.decay)
    best_loss = 1e10
    best_epoch = 0
    model.train()
    num_iter_per_epoch = len(training_generator)

    for epoch in range(start_epoch, opt.num_epoches):

        new_epoch = [int(kk) for kk in list(learning_rate_schedule.keys()) if int(kk) <= epoch][-1]

        if str(new_epoch) in learning_rate_schedule.keys():
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate_schedule[str(new_epoch)]
        for iter, batch in enumerate(training_generator):
            image, label = batch
            if torch.cuda.is_available():
                image = Variable(image.cuda(), requires_grad=True)
            else:
                image = Variable(image, requires_grad=True)
            optimizer.zero_grad()
            logits = model(image)
            loss, loss_coord, loss_conf, loss_cls = criterion(logits, label)
            loss.backward()
            optimizer.step()
            logger.info(
                "Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss:{:.4f} (Coord:{:.4f} Conf:{:.4f} Cls:{:.4f})".format(
                    epoch + 1,
                    opt.num_epoches,
                    iter + 1,
                    num_iter_per_epoch,
                    optimizer.param_groups[0]['lr'],
                    loss,
                    loss_coord,
                    loss_conf,
                    loss_cls))
            tb_logger.add_scalar('Train/Total_loss', loss, epoch * num_iter_per_epoch + iter)
            tb_logger.add_scalar('Train/Coordination_loss', loss_coord, epoch * num_iter_per_epoch + iter)
            tb_logger.add_scalar('Train/Confidence_loss', loss_conf, epoch * num_iter_per_epoch + iter)
            tb_logger.add_scalar('Train/Class_loss', loss_cls, epoch * num_iter_per_epoch + iter)
        if epoch % opt.test_interval == 0:
            model.eval()
            loss_ls = []
            loss_coord_ls = []
            loss_conf_ls = []
            loss_cls_ls = []
            for te_iter, te_batch in enumerate(test_generator):
                te_image, te_label = te_batch
                num_sample = len(te_label)
                if torch.cuda.is_available():
                    te_image = te_image.cuda()
                with torch.no_grad():
                    te_logits = model(te_image)
                    batch_loss, batch_loss_coord, batch_loss_conf, batch_loss_cls = criterion(te_logits, te_label)
                loss_ls.append(batch_loss * num_sample)
                loss_coord_ls.append(batch_loss_coord * num_sample)
                loss_conf_ls.append(batch_loss_conf * num_sample)
                loss_cls_ls.append(batch_loss_cls * num_sample)
            te_loss = sum(loss_ls) / val_set.__len__()
            te_coord_loss = sum(loss_coord_ls) / val_set.__len__()
            te_conf_loss = sum(loss_conf_ls) / val_set.__len__()
            te_cls_loss = sum(loss_cls_ls) / val_set.__len__()
            logger_val = logging.getLogger('val')  # validation logger
            logger_val.info(
                "# Validation # Epoch: {}/{}, Lr: {}, Loss:{:.2f} (Coord:{:.2f} Conf:{:.2f} Cls:{:.2f})".format(
                    epoch + 1,
                    opt.num_epoches,
                    optimizer.param_groups[0]['lr'],
                    te_loss,
                    te_coord_loss,
                    te_conf_loss,
                    te_cls_loss))
            tb_logger.add_scalar('Test/Total_loss', te_loss, epoch)
            tb_logger.add_scalar('Test/Coordination_loss', te_coord_loss, epoch)
            tb_logger.add_scalar('Test/Confidence_loss', te_conf_loss, epoch)
            tb_logger.add_scalar('Test/Class_loss', te_cls_loss, epoch)
            model.train()
            if te_loss + opt.es_min_delta < best_loss:
                best_loss = te_loss
                best_epoch = epoch
                save_filename = opt.model_name + '_{}.pth'.format(epoch)
                if isinstance(model, nn.DataParallel) or isinstance(model, DistributedDataParallel):
                    model = model.module
                state_dict = model.state_dict()
                for key, param in state_dict.items():
                    state_dict[key] = param.cpu()
                logger.info('Saving models and training states.')

                saved_path = os.path.join(opt.experiments_root, opt.model_name, 'trained_models')
                mkdir(saved_path)
                torch.save(state_dict, os.path.join(saved_path, save_filename))
                state = {'epoch': epoch, 'iter': te_iter}
                save_filename = opt.model_name + '_{}.state'.format(epoch)
                torch.save(state, os.path.join(saved_path, save_filename))

            # Early stopping
            if epoch - best_epoch > opt.es_patience > 0:
                logger_val.info("Stop training at epoch {}. The lowest loss achieved is {}".format(epoch, te_loss))
                break
    tb_logger.export_scalars_to_json(tensorboard_path + os.sep + "all_logs.json")
    tb_logger.close()


if __name__ == "__main__":
    opt = get_args()
    train(opt)
