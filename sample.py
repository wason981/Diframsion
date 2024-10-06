import torch
import argparse
torch.backends.cuda.matmul.allow_tf32=True
torch.backends.cudnn.allow_tf32=True
from omegaconf import OmegaConf
def main(args):
    config=OmegaConf.load(args.config)
    torch.set_grad_enabled(False)
    device="cuda" if torch.cuda.is_available() else "cpu"
    pretrained_t2v_model_path = './pretrained_models/watermark_remove_module.pt'
    unet=UNet3DConditionModelWaterMark.from_pretrained_2d(pretrained_model_path=)
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=str,required=True)
    args=parser.parse_args()
    main(args)