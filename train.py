from create_dataset import *
from utils import *
from SDCFusion import SDCFusion
from options import * 
from saver import Saver, resume
from time import time
from tqdm import tqdm
from optimizer import Optimizer
import datetime
from losses import * 
import math

def main():
    # parse options    
    parser = TrainOptions()
    opts = parser.parse()
    # define model, optimiser and scheduler
    device = torch.device("cuda:{}".format(opts.gpu) if torch.cuda.is_available() else "cpu")
    MPF_model = SDCFusion(opts.class_nb).to(device)
     # define dataset    
    train_dataset = MSRSData(opts, is_train=True)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=opts.batch_size,
        num_workers = opts.nThreads,
        shuffle=True)
    test_dataset = MSRSData(opts, is_train=False)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=30,
        num_workers = opts.nThreads,
        shuffle=False)    
    ## 先加载dataloader 计算每个epoch的的迭代步数 然后计算总的迭代步数
    ep_iter = len(train_loader)
    max_iter = opts.n_ep * ep_iter
    print('Training iter: {}'.format(max_iter))
    momentum = 0.9
    weight_decay = 5e-4
    lr_start = 1e-3
    # max_iter = 150000
    power = 0.9
    warmup_steps = 1000
    warmup_start_lr = 1e-5
    optimizer = Optimizer(
            model = MPF_model,
            lr0 = lr_start,
            momentum = momentum,
            wd = weight_decay,
            warmup_steps = warmup_steps,
            warmup_start_lr = warmup_start_lr,
            max_iter = max_iter,
            power = power)
    if opts.resume:
        MPF_model, optimizer.optim, ep, total_it = resume(MPF_model, optimizer.optim, opts.resume, device)
        optimizer = Optimizer(
            model = MPF_model,
            lr0 = lr_start,
            momentum = momentum,
            wd = weight_decay,
            warmup_steps = warmup_steps,
            warmup_start_lr = warmup_start_lr,
            max_iter = max_iter,
            power = power, 
            it=total_it)
        lr = optimizer.get_lr()
        print('lr:{}'.format(lr))
    else: 
        ep = -1
        total_it = 0
    ep += 1
    log_dir = os.path.join(opts.display_dir, 'logger', opts.name)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'log.txt')
    if os.path.exists(log_path):
        os.remove(log_path)
    logger = logger_config(log_path=log_path, logging_name='Timer')
    logger.info('Parameter: {:.6f}M'.format(count_parameters(MPF_model) / 1024 * 1024))
    
    # Train and evaluate multi-task network
    multi_task_trainer(train_loader,
                        test_loader,
                        MPF_model,
                        device,
                        optimizer,
                        opts,
                        logger,
                        ep,
                        total_it)
    
def multi_task_trainer(train_loader, test_loader, multi_task_model, device, optimizer, opt, logger, start_ep=0, total_it=0):
    total_epoch = opt.n_ep
    saver = Saver(opt)    
    ## 计算分割损失相关的设计
    mIoU = 0
    score_thres = 0.75
    ignore_idx = 255
    n_min = 16 * 256 * 256 // 16
    criteria = OhemCELoss(
        thresh=score_thres, n_min=n_min, device=device, ignore_lb=ignore_idx)    
    
    # criteria_fusion = Fusionloss(device=device)
    binary_class_weight = np.array([1.4548, 19.8962])    
    binary_class_weight = torch.tensor(binary_class_weight).float().to(device)    
    binary_class_weight = binary_class_weight.unsqueeze(0)
    binary_class_weight = binary_class_weight.unsqueeze(2)
    binary_class_weight = binary_class_weight.unsqueeze(2)
    lb_ignore = [255]
    if opt.resume:
        best_mIou = multi_task_tester(test_loader, multi_task_model, device, opt) - 0.02
    else:
        best_mIou = 0.0
    print('best mIoU: {:.4f}'.format(best_mIou))
    start = glob_st = time()
    for ep in range(start_ep, total_epoch): ## 每一个epoch 计算一次动态权重
        multi_task_model.train()        
        # Fusion_Criteria = Fusionloss(device=device)
        seg_metric = SegmentationMetric(opt.class_nb, device=device)          
        for it, (img_ir, img_vi, label) in enumerate(train_loader):
            total_it += 1
            img_ir = img_ir.to(device)
            img_vi = img_vi.to(device)
            Y_vi, Cb_vi, Cr_vi = RGB2YCrCb(img_vi)
            
            label = label.to(device)
            
            fused_img, seg_pred, fu_seg_pred, fused_img_color = multi_task_model(img_vi, img_ir)            
            optimizer.zero_grad()
            
            ############################Segmentation#############################
            seg_loss = Seg_loss(seg_pred, label, device, criteria)
            seg_results = torch.argmax(seg_pred, dim=1, keepdim=True) ## print(seg_result.shape())
            train_seg_loss = seg_loss # 源weight == 10  + 10 * bi_loss + 10 * bd_loss 
            
            ############################Validation#############################
            fu_seg_loss = Seg_loss(fu_seg_pred, label, device, criteria)
            fu_seg_results = torch.argmax(fu_seg_pred, dim=1, keepdim=True) ## print(seg_result.shape())
            train_fu_seg_loss = fu_seg_loss # 源weight == 10  + 10 * bi_loss + 10 * bd_loss 
            
            ## reconstruction-related loss            
            fusion_loss, int_loss, grad_loss = Fusion_loss(Y_vi, img_ir, fused_img, device=device)            

            ## mIoU-based semantic measurement weight ##
            lamd = 1 - 1 / (1 + exp(-10 * (2 * mIoU - 1)))
            
            train_loss = fusion_loss + lamd * train_seg_loss + (1 - lamd) * train_fu_seg_loss
            
            train_loss.backward()
            optimizer.step()
            seg_metric.addBatch(seg_results, label, lb_ignore)
        lr = optimizer.get_lr()
        mIoU = np.array(seg_metric.meanIntersectionOverUnion().item())
        Acc = np.array(seg_metric.pixelAccuracy().item())
        end = time()
        training_time, glob_t_intv = end - start, end - glob_st
        now_it = total_it + 1
        eta = int((total_epoch * len(train_loader) - now_it) * (glob_t_intv / (now_it)))
        eta = str(datetime.timedelta(seconds=eta))
        logger.info('ep: [{}/{}], learning rate: {:.6f}, time consuming: {:.2f}s, segmentation loss: {:.4f}, fusion loss: {:.4f}, validation loss: {:.4f}'.format(ep+1, total_epoch, lr, training_time, seg_loss, fusion_loss.item(), fu_seg_loss))
        logger.info('grad loss: [{:.4f}], int loss: [{:.4f}], segmentation loss: {:.4f}, validation loss: {:.4f}, mIou: {:.4f}, Acc: {:.4f}, Eta: {}, lamd: {:.4f}\n'.format(grad_loss, int_loss.item(), seg_loss, fu_seg_loss, mIoU, Acc, eta, lamd)) #, corr loss: [{:.4f}]
        start = time()
        ## save Visualization results
        if (ep + 1) % opt.img_save_freq == 0:
            input = [img_ir, img_vi, fused_img, fused_img_color, label]
            fused_rgb = fused_img_color#YCbCr2RGB(fused_img, vi_Cb, vi_Cr)
            vi_rgb = img_vi#YCbCr2RGB(re_vi, vi_Cb, vi_Cr)
            output = [img_ir, img_vi, fused_img, fused_img_color, seg_results]
            saver.write_img(ep, input, output)
        ## save model
        if (ep + 1) % opt.model_save_freq == 0:                       
            if (ep + 1) > 10:                             
                if (ep + 1) > 10:
                    saver.write_model(ep, total_it, multi_task_model, optimizer.optim, best_mIou, device, is_best=False)

def multi_task_tester(test_loader, multi_task_model, device, opts):
    multi_task_model.eval()
    test_bar= tqdm(test_loader)
    seg_metric = SegmentationMetric(opts.class_nb, device=device)
    lb_ignore = [255]
    ## define save dir
    with torch.no_grad():  # operations inside don't track history        
        for it, (img_ir, img_vi, label, img_names) in enumerate(test_bar):
            img_ir = img_ir.to(device)
            img_vi = img_vi.to(device)

            label = label.to(device)           
            fused_img, Seg_pred, fu_seg_pred, fused_img_color = multi_task_model(img_vi, img_ir)            
            seg_result = torch.argmax(Seg_pred, dim=1, keepdim=True) ## print(seg_result.shape())
            seg_metric.addBatch(seg_result, label, lb_ignore)        
    mIoU = np.array(seg_metric.meanIntersectionOverUnion().item())
    return mIoU

if __name__ == '__main__':
    main()
