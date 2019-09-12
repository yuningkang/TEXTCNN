import json
import  argparse
def data_path_config(file):
    with open(file,'r',encoding='UTF-8') as op:
        opt = json.load(op)
    return opt
def arg_config():
    #数据参数
    parser = argparse.ArgumentParser("Text classification")
    parser .add_argument("--epoch",type=int,default=5,help="iter number")
    parser.add_argument("--batch_size",type=int,default=32)
    #设备参数
    parser.add_argument("--device",type=str,default="cpu")
    #模型参数
    parser.add_argument("--hidden_size",type=int,default=200)
    parser.add_argument("--learning rate",type=float,default=3e-3)
    parser.add_argument("--lr",type =float,default=0.002)
    parser.add_argument("--weight_decay", type=float, default=1e-7)
    #嵌入层的drop
    parser.add_argument("--dropout_emb", type=float, default=0.5)
    parser.add_argument("--dropout_linear", type=float, default=0.5)
    #通用参数
    parser.add_argument("--cuda",type=int,default=1)
    args = parser.parse_args()
    return args
