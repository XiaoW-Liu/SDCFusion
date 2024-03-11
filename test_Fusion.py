from create_dataset import *
from utils import *
from SDCFusion import SDCFusion
from options import * 
from saver import resume, save_img_single
from tqdm import tqdm
from thop import profile

def main():
    # parse options    
    parser = TestOptions()
    opts = parser.parse()
    device = torch.device("cuda:0")
    
    Fusion_model = SDCFusion(opts.class_nb).to(device)
    Fusion_model = resume(Fusion_model, model_save_path=opts.resume, device=device, is_train=False)
    
    # define dataset    
    test_dataset = FusionData(opts)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=opts.batch_size,
        shuffle=False)
    
    # Train and evaluate multi-task network
    multi_task_tester(test_loader, Fusion_model, device, opts)
    
def multi_task_tester(test_loader, Fusion_model, device, opts):
    Fusion_model.eval()
    test_bar= tqdm(test_loader)
    ## define save dir
    Fusion_save_dir = os.path.join(opts.result_dir)
    print(Fusion_save_dir)
    os.makedirs(Fusion_save_dir, exist_ok=True)

    with torch.no_grad():  # operations inside don't track history
        for it, (img_ir, img_vi, img_names, widths, heights) in enumerate(test_bar):
            img_ir = img_ir.to(device)
            img_vi = img_vi.to(device)
            vi_Y, vi_Cb, vi_Cr = RGB2YCrCb(img_vi)
            vi_Y = vi_Y.to(device)
            vi_Cb = vi_Cb.to(device)
            vi_Cr = vi_Cr.to(device)       
            fused_img, seg_pred, _, _ = Fusion_model(img_vi, img_ir)  
            #fused_img = YCbCr2RGB(fused_img, vi_Cb, vi_Cr)
            for i in range(len(img_names)):
                img_name = img_names[i]
                fusion_save_name = os.path.join(Fusion_save_dir, img_name)
                save_img_single(fused_img[i, ::], fusion_save_name, widths[i], heights[i])
                test_bar.set_description('Image: {} '.format(img_name))

if __name__ == '__main__':
    main()
