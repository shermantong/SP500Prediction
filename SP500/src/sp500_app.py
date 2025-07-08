import gradio as gr
from model_predicting import *
from db_operation import *

def welcome_message():
    return "欢迎使用SP500指数预测系统！\n\n" \
           "本系统基于LSTM深度学习模型，能够根据历史数据预测SP500指数的未来走势。\n" \
           "请按照以下步骤操作：\n" \
           "1. 在下拉菜单中选择算法\n" \
           "2. 输入上一个交易日的SP500指数\n" \
           "3. 输入今天的SP500指数（建议在交易结束前5分钟输入）\n" \
           "4. 点击提交按钮\n" \
           "系统将显示预测的下一个交易日SP500指数和推荐决策（买入/卖出）。"

with gr.Blocks() as demo:
    gr.Markdown("# SP500指数预测系统")
    gr.Markdown(welcome_message())
    
    with gr.Row():
        algorithm = gr.Dropdown(label="算法选项", choices=["LSTM", "GRU", "RNN"], value="LSTM", interactive=True)
        # 创建两个数字输入框，分别用于获取用户输入的上一个交易日和今天的SP500指数
        # prev_day_index: 用于输入上一个交易日的SP500指数值
        # today_index: 用于输入当前交易日的SP500指数值（建议在交易结束前5分钟输入）
        prev_day_index = gr.Number(label="上一个交易日SP500指数")
        today_index = gr.Number(label="输入今天SP500指数")
    
    submit_btn = gr.Button("提交")
    
    with gr.Row():
        predicted_index = gr.Number(label="预测下一个交易日SP500指数", interactive=False)
        recommendation = gr.Textbox(label="推荐决策", interactive=False)
    
    def predict_and_recommend(algorithm, prev_day, today):
        # 简单预测逻辑：基于今日和昨日指数的变化趋势
        change = today - prev_day
        # 预测下一个交易日的指数（简单线性预测）
        predicted = today + change
        
        # 根据预测结果生成推荐
        if predicted > today:
            rec = "推荐买入"
        else:
            rec = "推荐卖出"
            
        return predicted, rec
    
    #暂时只测试UI，没有后端的预测代码
    def predict_and_save(algorithm, prev_day, today):
        # 保存上一个交易日数据
        save_prev(prev_day)
        
        # 进行预测
        predicted, rec = predict_and_recommend(algorithm, prev_day, today)
        
        # 保存今日数据和预测结果
        save_today(today, predicted)
        
        return predicted, rec
    
    submit_btn.click(
        fn=predict_and_save,
        inputs=[algorithm, prev_day_index, today_index],
        outputs=[predicted_index, recommendation]
    )

demo.launch()
