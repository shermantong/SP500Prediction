# 导入必要的库
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 定义模型类（需要与训练时相同）
class SP500Predictor(torch.nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1, scaler=None):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, output_size)
        self.scaler = scaler

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])
        return out

    def predict(self, x):
        # 对输入数据进行标准化
        x = self.scaler.transform(x)
        # 将numpy数组转换为PyTorch张量
        x = torch.tensor(x, dtype=torch.float32)
        # 增加batch维度并进行预测
        out = self(x[None, ...])
        # 将预测结果反标准化并返回标量值
        return self.scaler.inverse_transform(out.cpu().detach().numpy())[0, 0]

# 加载训练好的模型
def load_model(model_path):
    # 初始化模型
    model = SP500Predictor()
    # 加载模型参数
    model.load_state_dict(torch.load(model_path))
    # 设置模型为评估模式
    model.eval()
    return model

# 读取并准备数据
def prepare_data(data_path):
    # 读取CSV文件
    df = pd.read_csv(data_path)
    # 确保数据按日期排序
    df['observation_date'] = pd.to_datetime(df['observation_date'])
    df = df.sort_values('observation_date')
    # 获取SP500列的最后60个数据点
    sp500_data = df['SP500'].values.reshape(-1, 1)
    return sp500_data

# 使用模型进行预测
def predict_sp500(model, input_data):
    """
    使用训练好的模型进行预测
    参数：
    - model: 加载的模型
    - input_data: 输入数据，形状为(60, 1)的numpy数组
    返回：
    - 预测值
    """

    
    # 进行预测
    prediction = model.predict(input_data[:,None])
    return prediction

# 主程序
if __name__ == "__main__":
    import sys
    
    # 检查命令行参数
    if len(sys.argv) != 2:
        print("用法: python main.py <数据文件路径>")
        sys.exit(1)
    
    # 获取数据文件路径
    data_path = sys.argv[1]
    

    # 加载模型
    model = load_model('sp500_predictor.pth')
        
    # 准备数据
    input_data = prepare_data(data_path)
        
    # 进行预测
    prediction = predict_sp500(model, input_data)
    print(f"SP500指数预测结果: {prediction:.2f}")
    


