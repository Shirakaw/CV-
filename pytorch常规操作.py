### pytorch常用操作
####1.获取模型参数，梯度，名称
for name,param in model.named_parameters():
	if param.requires_grad:
		print(name.param.data)

for p in model.paramters():
	# p.requires_grad: bool
	#p.data: Tensor
for name,param in model.state_dict():
	# name: str
	# param: Tensor

####2. 加载/修改部分模型参数

checkpoint = torch.load(model_weight, map_location = lambda storage, loc: storage)
pretrained_dict = {'.'.join(k.split('.')[1:]) : v for k, v in checkpoint['state_dict'].items()}
model.load_state_dict(pretrained_dict)

# or
checkpoint = model.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)

#### 3.取消gard

with torch.no_grad():
	weight = (1.000001 - torch.exp(-loss)) ** self.gamma

#### 4.显示学习率

optimizer = optim.SGD(net.parameters(), lr = args.lr,momentum = 0.9,weight_decay = 5e-4)
optimizer.param_groups[0]['lr']

#### 5/array/tensor互转
#array->array
data = data.detch().cpu().numpy()

#### 6.保存tensor
np.save('target_lengths.npy', target_lengths.cpu().detach().numpy())
traget_lengths = torch.tensor(np.load(;target_length.npy), dtype = torch.int32)

#### 7.CTC loss
#对于pytorch的nn.CTCLoss(blank = len(dataset.chars),reduction = 'mean')
criterion = nn.CTCLoss(blank=4652, reduction = 'mean')

target_lengths = torch.tensor([5],dtype = torch.int32)

criterion(log_probs.log_softmax(2), targets,input_lengths,target_lengths)tensor(5.6648)

targets = torch.tensor([[4642, 4642, 4647, 4643,4648,4640,4639,4639]],dtype=torch.int32)

target_lengths = torch.tensor([7],dtype = torch.int32)

criterion(log_probs.log_softmax(2),targets.input_lengths,targets,input_length,target_lengths)
tensor(6.6101)

target_lenghts = torch.tensor([8],dtype = torch.int32)

criterion(log_probs.log_softmax(2),targets, input_lengths,target_lengths)
tensor(inf)
随着target_lengths与input_lenghths的接近,loss趋近inf

#### 8,tensor转置

data.permute(1,2,0).detach().cpu().numpy()

#### 9, C++中多个tensor输出
auto output = module.forward({tensor_image}).toTuple();
torch::Tensor output = outputs->elements()[0].toTensor().squeeze(0).to(torch::kCPU);
torch::Tensor t_feature = outputs->elements()[1].toTensor().squeeze(0).to(torch::kCPU);

#### 10,torch.is_grad_enabled(),model.training.torch.no_grad()的联系
torch.is_grad_enabled():设置之后，不再计算梯度
model.training:由model.eval()/train()控制
with torch.no_grad():设置部分变量不计算梯度

#### 11,tensor.detach和tensor.data的区别
#tensor.detach返回一个新的tensor，从当前计算图中分离下来的，但是仍指向原变量的存放位置，不同之处只是requires_grad为false，得到的这个tensor永远不需要计算其梯度，不具有grad。即使之后重新将它的requires_grad置为true，它也不会具有梯度grad。这样我们就会继续使用这个新的tensor进行计算，后面当我们进行反向传播时，到该调用detach()的tensor就会停止，不能再继续向前进行传播。
#tensor.data的不同在于.data的修改不会被autograd追踪，这样当进行backward()时它不会报错，会得到一个错误的backward值。

#### 12，不同层设置不同的学习率
#划分不同学习率
def para_group(model, lr):
    parameters_1x = [param for name, param in model.named_parameters() if name not in ['fc.weight', 'fc.bias']]
    optimizer = torch.optim.SGD([{'params': parameters_1x},
                                 {'params': model.fc.parameters(), 'lr': lr * 10}],
                                 lr, momentum=0.9, weight_decay=1e-4)
    return optimizer

#### 划分数据集
#方式一：
transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize((0.5,), (0.5,))])
full_dataset = MNIST(root = './data', train = train, transform = transform, download=True)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
#方法二
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import DataLoader

def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets