# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import time

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils.utils import AverageMeter, distributed_all_gather

from monai.data import decollate_batch

# 使用 matplotlib 绘图
import matplotlib.pyplot as plt

def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, acc_func, args, post_label=None, post_pred=None):
    model.train()
    start_time = time.time()
    trainr_loss = AverageMeter()
    trainr_acc = AverageMeter() # !!!!!!!!!!
    for idx, batch_data in enumerate(loader):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]
        data, target = data.cuda(args.rank), target.cuda(args.rank)

        # 清空梯度
        for param in model.parameters():
            param.grad = None

        # 前向传播与损失计算
        with autocast(enabled=args.amp):
            logits = model(data)  # 前向传播，得到模型的预测输出
            loss = loss_func(logits, target) # 计算预测结果和真实标签之间的损失

        # 计算准确率 !!!!!!!!!!!!!!!!!!!
        trainr_labels_list = decollate_batch(target)  # 使用 decollate_batch 将批次的目标标签拆分成单个样本,
        trainr_labels_convert = [post_label(trainr_label_tensor) for trainr_label_tensor in
                              trainr_labels_list]  # 对每个标签应用 post_label 后处理函数。
        trainr_outputs_list = decollate_batch(logits)  # 使用 decollate_batch 将模型输出 logits 拆分成单个样本。
        trainr_output_convert = [post_pred(trainr_pred_tensor) for trainr_pred_tensor in
                              trainr_outputs_list]  # 对每个预测值应用 post_pred 后处理函数。

        acc_func.reset()  # 重置准确率计算函数 acc_func，准备进行新的计算。
        acc_func(y_pred=trainr_output_convert,
                 y=trainr_labels_convert)  # 使用后处理后的预测值 val_output_convert 和标签 val_labels_convert 更新准确率计算函数。
        acc, not_nans = acc_func.aggregate()  # 从 acc_func 中获取计算出的准确率 acc 和有效样本数量 not_nans。
        acc = acc.cuda(args.rank)  # 将准确率 acc 移动到指定的 GPU 上。

        # 反向传播与优化
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # 更新损失和准确率
        if args.distributed:
            loss_list = distributed_all_gather([loss.item()], out_numpy=True, is_valid=idx < loader.sampler.valid_length) # 收集损失值
            trainr_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
            ) #每个设备处理的批次大小。 设备总数（例如，GPU 数量）。

            acc_list, not_nans_list = distributed_all_gather([acc, not_nans], out_numpy=True,
                                                             is_valid=idx < loader.sampler.valid_length)  # # 使用 distributed_all_gather 汇总所有设备上的准确率和有效样本数量
            for al, nl in zip(acc_list, not_nans_list):
                trainr_acc.update(al, n=nl) # 对于每个设备上的准确率和有效样本数量，更新 run_acc，即更新平均准确率
        else:
            trainr_loss.update(loss.item(), n=args.batch_size)
            trainr_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy()) ###### 如果没有启用分布式训练，则直接更新 run_acc，计算当前设备的准确率。

        # 打印训练进度
        if args.rank == 0:
            avg_acc = np.mean(trainr_acc.avg)
            print(
                "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                "loss: {:.4f}".format(trainr_loss.avg),
                "acc: {:.4f}". format(avg_acc),
                "time {:.2f}s".format(time.time() - start_time),
            )
        start_time = time.time()

    for param in model.parameters():
        param.grad = None
    return trainr_loss.avg, trainr_acc.avg


# 验证过程中的损失和准确率更新
def val_epoch(model, loader, epoch, acc_func, loss_func, args, model_inferer=None, post_label=None, post_pred=None):
    model.eval()
    valr_loss = AverageMeter()  # 验证损失 !!!!!!!!!!!!!!!
    valr_acc = AverageMeter()  # 验证准确率
    start_time = time.time()

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]
            data, target = data.cuda(args.rank), target.cuda(args.rank)

            # 前向传播与损失计算
            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    logits = model_inferer(data)
                else:
                    logits = model(data)
                loss = loss_func(logits, target)

            if not logits.is_cuda:
                target = target.cpu()

            # 计算准确率
            val_labels_list = decollate_batch(target)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_convert)
            acc, not_nans = acc_func.aggregate()
            acc = acc.cuda(args.rank)

            # 更新验证准确率和损失
            if args.distributed:
                # 收集准确率和有效样本数
                acc_list, not_nans_list = distributed_all_gather([acc, not_nans], out_numpy=True,
                                                                 is_valid=idx < loader.sampler.valid_length)
                # 收集损失值
                loss_list = distributed_all_gather([loss.item()], out_numpy=True,
                                                   is_valid=idx < loader.sampler.valid_length)

                # 更新验证准确率
                for al, nl in zip(acc_list, not_nans_list):
                    valr_acc.update(al, n=nl)

                # 使用 np.stack 和 np.mean 计算损失的平均值
                loss_mean = np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0)
                valr_loss.update(loss_mean, n=args.batch_size * args.world_size)  # 乘以设备总数

            else:
                # 非分布式情况，直接更新损失和准确率
                valr_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())
                valr_loss.update(loss.item(), n=args.batch_size)

            # 打印验证进度
            if args.rank == 0:
                avg_acc = np.mean(valr_acc.avg)
                print(
                    "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                    "loss: {:.4f}".format(valr_loss.avg),
                    "acc: {:.4f}". format(avg_acc),
                    "time {:.2f}s".format(time.time() - start_time),
                )

    for param in model.parameters():
        param.grad = None
    return valr_loss.avg, valr_acc.avg


def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    args,
    model_inferer=None,
    scheduler=None,
    start_epoch=0,
    post_label=None,
    post_pred=None,
):
    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0:
            print("Writing Tensorboard logs to ", args.logdir)
    scaler = None
    if args.amp:
        scaler = GradScaler() # 作用: 如果启用了自动混合精度训练 (args.amp)，则初始化 torch.cuda.amp.GradScaler，用于缩放梯度，减少显存占用并加速训练。
    val_acc_max = 0.0

    ######### 初始化损失和准确率记录列表
    train_loss_list = []  # 用于记录训练损失
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []     # 用于记录验证准确率

    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier() # 如果是分布式训练（args.distributed），则设置 train_loader.sampler 的 epoch，并同步所有进程。

        print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()

        # 训练过程
        train_loss, train_acc, = train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, acc_func=acc_func, args=args, post_label=post_label, post_pred=post_pred
        )
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        if args.rank == 0:
            avg_train_acc = np.mean(train_acc)
            print(
                "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                "training loss: {:.4f}".format(train_loss),
                "training accuracy: {:.4f}".format(avg_train_acc),
                "time {:.2f}s".format(time.time() - epoch_time),
            )
        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)
            writer.add_scalar("train_acc", train_acc, epoch)
        b_new_best = False

        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()

            val_loss, val_acc = val_epoch(
                model, val_loader, epoch=epoch, acc_func=acc_func, loss_func=loss_func, args=args,
                model_inferer=model_inferer, post_label=post_label, post_pred=post_pred
            )

            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)  # 记录验证准确率

            if args.rank == 0:
                avg_val_acc = np.mean(val_acc)
                print(
                    "Final validation  {}/{}".format(epoch, args.max_epochs - 1),
                    "validation loss: {:.4f}".format(val_loss),
                    "validation accuracy: {:.4f}".format(avg_val_acc),
                    "time {:.2f}s".format(time.time() - epoch_time),
                )
                if writer is not None:
                    writer.add_scalar("val_acc", val_acc, epoch)
                    writer.add_scalar("val_loss", val_loss, epoch)

                if val_acc > val_acc_max:
                    print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, avg_val_acc))
                    val_acc_max = avg_val_acc
                    b_new_best = True
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(
                            model, epoch, args, best_acc=val_acc_max, optimizer=optimizer, scheduler=scheduler
                        )
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="model_final.pt")
                if b_new_best:
                    print("Copying to model.pt new best model!!!!")
                    shutil.copyfile(os.path.join(args.logdir, "model_final.pt"), os.path.join(args.logdir, "model.pt"))

        if scheduler is not None:
            scheduler.step()
    print("Training Finished!, Best Accuracy: ", val_acc_max)

    # 创建两个图形
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # 绘制 Loss 图
    axs[0].plot(np.arange(len(train_loss_list)), train_loss_list, label="train loss")
    axs[0].plot(np.arange(1, len(val_loss_list) * 50, 50), val_loss_list, label="validation loss")
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Model Loss')
    axs[0].legend()

    # 绘制 Accuracy 图
    axs[1].plot(np.arange(len(train_acc_list)), train_acc_list, label="train acc")
    axs[1].plot(np.arange(1, len(val_acc_list) * 50, 50), val_acc_list, label="validation acc")
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_title('Model Accuracy')
    axs[1].legend()

    # 保存到本地文件夹
    plt.tight_layout()
    plt.savefig('./runs/train/loss_accuracy_plots.png')  # 请替换成实际保存路径

    return val_acc_max
