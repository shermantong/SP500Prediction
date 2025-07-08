import datetime
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

# 定义模型
class SP500Predictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1, scaler=None):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.scaler = scaler  # 添加标准化器

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])
        return out

    def predict(self, x):
        """
        预测函数，用于单个样本的预测
        参数：
        - x: 输入数据，形状为(sequence_length, 1)
        返回：
        - 反标准化后的预测值
        """
        # 对输入数据进行标准化
        x = self.scaler.transform(x)
        # 将numpy数组转换为PyTorch张量
        x = torch.tensor(x, dtype=torch.float32)
        # 增加batch维度并进行预测
        out = self(x[None, ...])
        # 将预测结果反标准化并返回标量值
        # 使用detach()将张量从计算图中分离，避免梯度计算
        # 然后转换为numpy数组进行反标准化
        return self.scaler.inverse_transform(out.cpu().detach().numpy())[0, 0]



def train(data_path, batch_size=100):
    """
    训练模型函数
    参数：
    - data_path: 数据文件路径
    - batch_size: 批量大小，默认为100
    返回：
    - 训练好的模型
    """
    # 创建数据加载器
    train_loader, dev_loader, scaler = create_feeder(data_path, batch_size)

    # 初始化模型、损失函数和优化器
    model = SP500Predictor(scaler=scaler)
    # model.scaler = scaler  # 将scaler传入模型
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练过程
    num_epochs = 100
    
    def train_step(model, criterion, optimizer, loader):
        """单次训练步骤"""
        model.train()
        total_loss = 0
        for inputs, targets in loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.inference_mode()
    def eval_step(model, criterion, loader):
        """验证步骤"""
        model.eval()
        total_loss = 0
        for inputs, targets in loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
        return total_loss / len(loader)

    for epoch in range(num_epochs):
        train_loss = train_step(model, criterion, optimizer, train_loader)
        val_loss = eval_step(model, criterion, dev_loader)
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    return model

def create_feeder(data_path, batch_size):
    """
    创建数据加载器
    参数：
    - data_path: 数据文件路径
    - batch_size: 批量大小
    返回：
    - train_set: 训练集数据加载器
    - devset: 验证集数据加载器
    - scaler: 数据标准化器
    """
    # 读取数据并预处理
    df = load_data(data_path)
    data = df['SP500'].values.reshape(-1, 1)

    # 划分训练集和验证集
    train_size = int(len(data) * 0.8)  # 80% 训练，20% 验证
    train_data, dev_data = data[:train_size], data[train_size:]

    # 数据标准化
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    dev_scaled = scaler.transform(dev_data)

    # 创建序列数据
    sequence_length = 45
    def create_sequences(data):
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i+sequence_length])
            y.append(data[i+sequence_length])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_scaled)
    X_dev, y_dev = create_sequences(dev_scaled)


    def create_loader(X, y, batch_size, shuffle):
        """创建数据加载器
        参数：
        - X: 特征数据
        - y: 标签数据
        - batch_size: 批量大小
        - shuffle: 是否打乱数据
        返回：
        - DataLoader对象
        """
        # 将numpy数组转换为PyTorch张量
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        # 创建TensorDataset
        dataset = TensorDataset(X_tensor, y_tensor)
        # 创建并返回DataLoader
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    # 使用create_loader函数创建训练集和验证集加载器
    train_set = create_loader(X_train, y_train, batch_size, shuffle=True)
    dev_set = create_loader(X_dev, y_dev, batch_size, shuffle=False)

    return train_set, dev_set, scaler

def load_data(path):
    df = pd.read_csv(path)
    df['observation_date'] = pd.to_datetime(df['observation_date'])
    # 按观察日期排序数据，确保时间序列按时间顺序排列
    df = df.sort_values('observation_date')
    # 删除SP500列中值为NaN的行，确保数据完整性
    df = df.dropna(subset=['SP500'])
    return df

if __name__ == '__main__':
    import sys
    # 检查命令行参数数量
    if len(sys.argv) != 3:
        print("用法: python model_training.py <训练数据文件> <测试数据文件>")
        sys.exit(1)
    
    # 从命令行获取文件路径
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    
    # 训练模型
    model = train(train_file)
    
    # 定义测试函数
    def test(model, test_path):
        model.eval()
        # 读取测试数据
        tf = load_data(test_path)
        # 获取SP500列数据
        test_data = tf['SP500'].values
        # 使用模型进行预测
        prediction = model.predict(test_data[:, None])
        return prediction
    
    # 测试模型
    test_result = test(model, test_file)
    print(f'测试结果: {test_result:.2f}')

# 保存训练好的模型
torch.save(model.state_dict(), 'sp500_predictor.pth')


# 解释：这里的model size是指模型的参数数量和结构大小，
# 例如LSTM层的hidden_size为50，num_layers为2。
# 而batch_size是指每次训练迭代中使用的样本数量。
# 这两个概念不同，model size与模型的复杂度和容量有关，
# 而batch_size影响训练的效率和稳定性。

# 如何运行和使用模型
# 1. 确保安装了必要的库：torch, pandas。
# 2. 运行此脚本以训练模型并保存为 'sp500_predictor.pth'。
# 3. 使用模型时，加载模型参数：
#    model = SP500Predictor()
#    model.load_state_dict(torch.load('sp500_predictor.pth'))
# 4. 将模型设置为评估模式：model.eval()
# 5. 使用模型进行预测时，输入数据需转换为张量格式，并确保形状与训练时一致。