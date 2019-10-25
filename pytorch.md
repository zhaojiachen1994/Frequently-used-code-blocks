# 统计网络的参数数量
```python
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}
```


# LSTM的输入输出形状
- 参数：
    -input_size：The number of expected features in the input x
    -hidden_size: The number of features in the hidden state h
- Shape:
    - input shape: **(seq_len, batch, input_size)**
    - output shape: **(seq_len, batch, num_directions * hidden_size)**
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

# LSTMCell 的输入输出
- 输入输出
    - Inputs: input, (h_0, c_0)
    - Outputs: (h_1, c_1)

- 参数：
    - input_size：The number of expected features in the input x
    - hidden_size: The number of features in the hidden state h
- Shape:
    - input shape: **(batch, input_size)**
    - output shape: **h_1 of shape (batch, hidden_size)  c_1 of shape (batch, hidden_size) **

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

    
