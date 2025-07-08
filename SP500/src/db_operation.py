import sqlite3
import pandas as pd
from datetime import datetime

# 数据库配置
DB_PATH = 'sp500.db'

def create_connection():
    """创建并返回数据库连接"""
    conn = sqlite3.connect(DB_PATH)
    return conn

def data_loader():
    """
    从数据库加载最新的80个有效交易日数据
    返回：包含sp500_index列的DataFrame
    """
    conn = create_connection()
    
    # 查询最新的80个有效交易日数据
    query = '''
        SELECT date, sp500_index 
        FROM sp500_data 
        WHERE sp500_index IS NOT NULL
        ORDER BY date DESC 
        LIMIT 80
    '''
    df = pd.read_sql(query, conn)
    conn.close()
    
    # 确保数据按日期升序排列
    df = df.sort_values('date')
    return df

def save_prev(prev_day_index):
    """
    保存上一个交易日SP500指数到数据库
    参数：
    - prev_day_index: 上一个交易日SP500指数
    """
    if prev_day_index is None:
        return
        
    conn = create_connection()
    cursor = conn.cursor()
    
    # 获取上一个交易日日期
    prev_day = (datetime.now() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    
    # 更新上一个交易日的数据
    cursor.execute('''
        UPDATE sp500_data 
        SET sp500_index = ?
        WHERE date = ?
    ''', (prev_day_index, prev_day))
    
    conn.commit()
    conn.close()

def save_today(today_index, predicted_index):
    """
    保存今天SP500指数和预测的下一个交易日SP500指数到数据库
    参数：
    - today_index: 当日SP500指数
    - predicted_index: 预测的下一个交易日SP500指数
    """
    conn = create_connection()
    cursor = conn.cursor()
    
    # 获取当前日期
    today = datetime.now().strftime('%Y-%m-%d')
    
    # 插入或更新今日数据
    cursor.execute('''
        INSERT OR REPLACE INTO sp500_data 
        (date, sp500_index, sp500_index_5min, predicted_sp500_index) 
        VALUES (?, ?, ?, ?)
    ''', (today, today_index, today_index, predicted_index))
    
    conn.commit()
    conn.close()

