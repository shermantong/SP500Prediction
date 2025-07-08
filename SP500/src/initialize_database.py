import sqlite3
import pandas as pd

# 连接到SQLite数据库
conn = sqlite3.connect('sp500.db')
cursor = conn.cursor()

# 创建表
cursor.execute('''
CREATE TABLE IF NOT EXISTS sp500_data (
    date TEXT PRIMARY KEY,
    sp500_index REAL,
    sp500_index_5min REAL,
    predicted_sp500_index REAL
)
''')

# 从命令行参数获取CSV文件路径并读取数据
import sys
if len(sys.argv) < 2:
    print("请提供CSV文件路径作为命令行参数")
    sys.exit(1)
data = pd.read_csv(sys.argv[1])

# 插入数据
for index, row in data.iterrows():
    cursor.execute('''
    INSERT OR REPLACE INTO sp500_data (date, sp500_index)
    VALUES (?, ?)
    ''', (row['observation_date'], row['SP500']))

conn.commit()
conn.close() 