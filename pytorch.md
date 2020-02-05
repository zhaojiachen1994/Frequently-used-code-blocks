## Device configuration
```python
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

## 统计网络的参数数量
```python
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}
```

## 查看网络参数值
```python
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, '\t', param.size())
```


## LSTM的输入输出形状
- 参数：
    - input_size：The number of expected features in the input x
    - hidden_size: The number of features in the hidden state h
- Shape:
    - input shape: **(seq_len, batch, input_size)**
    - output shape: **(seq_len, batch, num_directions x hidden_size)**
    - h_o, c_0, h_n, c_n shape: (num_layers * num_directions, batch, hidden_size)
```python
rnn = nn.LSTM(input_size=10, hidden_size=20, num_layers=2,
              bias=False, batch_first=False, dropout=0, bidirectional=False)
input = torch.randn(5, 3, 10)   #[seq_len, batch_size, input_size]
h0 = torch.randn(2, 3, 20)  #initial hidden state, [num_layers*num_directions, batch_size, hidden_size]
c0 = torch.randn(2, 3, 20)  #initial cell state, [num_layers*num_directions, batch_size, hidden_size]
output, (hn, cn) = rnn(input, (h0, c0))
print(output.shape) # [5, 3, 20]
print(hn.shape) # [2, 3, 20]
```

## LSTMCell 的输入输出
    - 输入输出
        - Inputs: input, (h_0, c_0)
        - Outputs: (h_1, c_1)

    - 参数：
        - input_size：The number of expected features in the input x
        - hidden_size: The number of features in the hidden state h
    - Shape:
        - input shape: **(batch, input_size)**
        - output shape: 
    
        **h_1 of shape (batch, hidden_size)**
        
        **c_1 of shape (batch, hidden_size)**

```python 
rnn = nn.LSTMCell(10, 20)   # 10 is input_size, 20 is hidden_size)
input = torch.randn(6, 3, 10)   # (seq_len, batch_size, input_size)
hx = torch.randn(3, 20) # [batch_size, hidden_size]
cx = torch.randn(3, 20) # [batch_size, hidden_size]
output = []
#   循环输入每个步长
for i in range(6):
    hx, cx = rnn(input[i], (hx, cx))
    output.append(hx)
print(len(output))  # 6
```
## Pytorch 如何保存模型加载模型

Reference: https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training
    
   - torch.save()
    
```pyton
def saveRNN(model, config):
    #:param model: A pytorch nn model
    #:param config: The parameters needed for the code
    #:return:
    checkpoint = {
        'config': config,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': model.optimizer.state_dict()}
    torch.save(checkpoint, config.pathCheckpoint)
    # parser.add_argument('--pathCheckpoint', type=str, default = './checkpoint/RNN.ckpt')
```
 
   - torch.laod()

```python
def saveRNN(model, config):
    #:param model: A pytorch nn model
    #:param config: The parameters needed for the code
    #:return:
    checkpoint = {
        'config': config,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': model.optimizer.state_dict()}
    torch.save(checkpoint, config.pathCheckpoint)
    # parser.add_argument('--pathCheckpoint', type=str, default = './checkpoint/RNN.ckpt')
```

## 设置随机种子，固定网络的初始权重
```python
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)
# 预处理数据以及训练模型
# ...
# ...
```

<details> 
    <summary><strong>   高斯法初始化网络权重   </strong></summary>

```python
      def init_weights(self):#定义在model类中，执行在初始化最后阶段。
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)
```
</details>

-----------------------------------------------------------------------------------------------------------------------------------

<details><summary><strong>   Array to dataloader  </strong></summary><blockquote>

```python
    from torch.utils.data import DataLoader
    X = np.linspace(1,1200, 1200).reshape([600,2])
    data_loader = DataLoader(dataset=X, batch_size=30, shuffle=False, drop_last=False)
    for x in data_loader:
        print(x.shape)
    print(len(data_loader))
```

</blockquote></details>

-----------------------------------------------------------------------------------------------------------------------------------

<details><summary><strong>   样本和标签两个array构造DataLoader  </strong></summary><blockquote>
    
 ```python
    from torch.utils.data import DataLoader, Dataset, TensorDataset
    X = np.linspace(1,30, 30).reshape([15, 2])
    y = np.linspace(1,15, 15)
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()
    data_loader = DataLoader(dataset=TensorDataset(X, y), batch_size=5, shuffle=False, drop_last=False)
    for [x_batch, y_batch] in data_loader:
        print(x_batch)
        print(y_batch)
    print(len(data_loader))
 ```

</blockquote></details>

-----------------------------------------------------------------------------------------------------------------------------------

<details>
<summary><strong>   tensorboardX 可视化(远程) </strong></summary>

- Install packages and activate the pytorch environment

    conda install -c conda-forge tensorboard
    
    conda install -c conda-forge tensorboardx
 
    conda activate pytorch

- Codes

    from tensorboardX import SummaryWriter (or from torch.utils.tensorboard import SummaryWriter 
    
    writer = SummaryWriter('/tmp/runs') # the path should be difined as exact this 
    
    writer.add_image('four_fashion_mnist_images', img_grid)

- remote host (in terminal)

    tensorboard -- logdir=/tmp/runs --port=8887
    
- local host (in window terminal)

    ssh -N -L localhost:8888:localhost:8887 zjc@131.128.54.107
    
    open brower with localhost:8888
 


</details>

-----------------------------------------------------------------------------------------------------------------------------------

<details>
<summary><strong>   Pytorch + tensorboardX example  </strong></summary>

 ```python
import numpy as np
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from tensorboardX import SummaryWriter


class AutoEncoderModule(nn.Module):#, PyTorchUtils
    def __init__(self, n_features: int, hidden_size: int, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        input_length = n_features
        # creates powers of two between eight and the next smaller power from the input_length
        dec_steps = 2 ** np.arange(max(np.ceil(np.log2(hidden_size)), 2), np.log2(input_length))[1:]
        dec_setup = np.concatenate([[hidden_size], dec_steps.repeat(2), [input_length]])
        enc_setup = dec_setup[::-1]
        layers = np.array([[nn.Linear(int(a), int(b)), nn.Tanh()] for a, b in enc_setup.reshape(-1, 2)]).flatten()[:-1]
        self._encoder = nn.Sequential(*layers)
        layers = np.array([[nn.Linear(int(a), int(b)), nn.Tanh()] for a, b in dec_setup.reshape(-1, 2)]).flatten()[:-1]
        self._decoder = nn.Sequential(*layers)

    def forward(self, input_batch=None, target_batch=None, return_latent: bool = False):
        '''
        input shape is [batch_size, sequence_length, n_feature]
        '''
        enc = self._encoder(input_batch)
        dec = self._decoder(enc)
        loss = (target_batch - dec).view([-1, 2])
        loss1=loss[:,0]+Variable(torch.Tensor([2.]))    # operate with a constant
        loss2=torch.log(loss[:,1]) # log computing
        loss3=loss1+loss2 # slice a tensor and add its elements
        return loss3, loss4


def checkADE():
    seed=0
    setup_seed(seed)
    n_features = 2
    n_samples = 1
    writer = SummaryWriter('/tmp/runs1')
    input = np.random.rand(n_samples, n_features)
    target = np.random.rand(n_samples, n_features)
    input = torch.from_numpy(input).float()
    target = torch.from_numpy(target).float()
    model = AutoEncoderModule(n_features=n_features, hidden_size=1, seed=seed)
    output = model.forward(input, target)

    writer.add_graph(model, (input, target))
    writer.close()

if __name__ == "__main__":
    checkADE()
 ```

</details>

<div align=left><img src ="https://github.com/zhaojiachen1994/Frequently-used-code-blocks/blob/master/Figures/tensorboardexample.PNG" width="200" height="120"/></div>

-----------------------------------------------------------------------------------------------------------------------------------

## 调整learning rate
https://pytorch.org/docs/stable/optim.html
https://zhuanlan.zhihu.com/p/39020473
