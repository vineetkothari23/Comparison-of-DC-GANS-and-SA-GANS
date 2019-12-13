import argparse

def str2bool(v):
    return v.lower() in ('true')

def sagan_parameters():

    parser = argparse.ArgumentParser()
	
    parser.add_argument('--root_dir', type=str, default="")
    parser.add_argument('--version', type=str, default="1.0")
	parser.add_argument('--model', type=str, default="sagan")
	parser.add_argument('--dataset', type=str, default="cifar")

    #train setting
	parser.add_argument('--adv_loss', type=str, default="wgan-gp")
	parser.add_argument('--g_num', type=int, default=5)
	parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--use_tensorboard', type=boolean, default=True)
    parser.add_argument('--d_iters', type=int, default=10)
	parser.add_argument('--train', type=boolean, default=False)
	parser.add_argument('--parallel', type=str, default=False)

    #input
    parser.add_argument('--imsize', type=int, default=64)
    parser.add_argument('--inp_height', type=int, default=256)
    parser.add_argument('--z_dim', type=int, default=100)
	parser.add_argument('--label_dim', type=int, default=10)


    # Model hyper-parameters
    parser.add_argument('--g_conv_dim', type=int, default=64)
    parser.add_argument('--d_conv_dim', type=int, default=64)
    parser.add_argument('--lr_decay', type=float, default=0.95)
    parser.add_argument('--lrD', type=float, default=0.02)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--lambda_gp', type=float, default=10)
    
    # # using pretrained
    # parser.add_argument('--pretrained_model', type=int, default=34375)


    return parser.parse_args()
	
def dcgan_parameters():
	parser = argparse.ArgumentParser()
	#root_dir="/content/drive/My Drive/Fall 2019/Deep Learning CSE 676/Projects/1/DC Gans/"
	parser.add_argument('--root_dir', type=str, default="")
	root_dir=""
	#output images
	parser.add_argument('--output_dir', type=str, default=root_dir+"output/epoch/")
	#models
	parser.add_argument('--model_dir', type=str, default=root_dir)
	#parameters
	parser.add_argument('--batch_size', type=int, default=4)
	#input
	parser.add_argument('--data_dir', type=str, default="data")
	parser.add_argument('--inp_width', type=int, default=32)
	parser.add_argument('--inp_height', type=int, default=32)
	parser.add_argument('--inp_channels', type=int, default=3)
	parser.add_argument('--nc', type=int, default=3)
	parser.add_argument('--nz', type=int, default=100)
	parser.add_argument('--ngf', type=int, default=2)
	parser.add_argument('--ndf', type=int, default=2)
	return parser.parse_args()

