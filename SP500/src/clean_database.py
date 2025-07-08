import sqlite3
import sys
import argparse

def remove_null_records(conn):
    """删除包含空值的记录"""
    cursor = conn.cursor()
    cursor.execute("DELETE FROM sp500_data WHERE sp500_index IS NULL")
    print("已删除包含空值的记录")

def remove_duplicate_records(conn):
    """删除重复的记录，保留最新的一条"""
    cursor = conn.cursor()
    cursor.execute('''
    DELETE FROM sp500_data
    WHERE rowid NOT IN (
        SELECT MIN(rowid)
        FROM sp500_data
        GROUP BY date
    )
    ''')
    print("已删除重复记录")

def validate_date_format(conn):
    """验证日期格式并删除无效记录"""
    cursor = conn.cursor()
    cursor.execute('''
    DELETE FROM sp500_data
    WHERE date NOT LIKE '____-__-__'
    ''')
    print("已删除无效日期格式的记录")

def sort_data_by_date(conn):
    """按日期排序数据"""
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS temp_sp500_data AS
    SELECT * FROM sp500_data ORDER BY date
    ''')
    cursor.execute('DROP TABLE sp500_data')
    cursor.execute('ALTER TABLE temp_sp500_data RENAME TO sp500_data')
    print("数据已按日期排序")

def clear_all_data(conn):
    """清空数据库中的所有数据"""
    cursor = conn.cursor()
    cursor.execute("DELETE FROM sp500_data")
    print("已清空数据库中的所有数据")

def clean_database(db_path='sp500.db', operations=None):
    """
    根据指定的操作清理SP500数据库
    :param db_path: 数据库文件路径
    :param operations: 要执行的操作列表，如果为None则执行所有操作
    """
    try:
        # 连接到数据库
        conn = sqlite3.connect(db_path)
        
        # 定义所有可用操作
        all_operations = {
            'remove_null': remove_null_records,
            'remove_duplicate': remove_duplicate_records,
            'validate_date': validate_date_format,
            'sort_data': sort_data_by_date,
            'clear_all': clear_all_data
        }

        # 如果没有指定操作，则执行所有操作（除了clear_all）
        if operations is None:
            operations = [op for op in all_operations.keys() if op != 'clear_all']

        # 执行选定的操作
        for op in operations:
            if op in all_operations:
                all_operations[op](conn)
            else:
                print(f"警告：未知操作 '{op}'，已跳过")

        # 提交更改
        conn.commit()
        print("数据库清理完成！")
        
    except sqlite3.Error as e:
        print(f"数据库清理过程中发生错误: {str(e)}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="SP500数据库清理工具")
    parser.add_argument('--db', default='sp500.db', help='数据库文件路径')
    parser.add_argument('--ops', nargs='+', 
                       choices=['remove_null', 'remove_duplicate', 'validate_date', 'sort_data', 'clear_all'],
                       help='指定要执行的操作（可多选）')
    
    args = parser.parse_args()
    
    # 执行清理操作
    clean_database(db_path=args.db, operations=args.ops)
