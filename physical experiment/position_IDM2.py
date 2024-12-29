import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置全局字体为 Times New Roman
rcParams['font.family'] = 'Times New Roman'

# Define moving average function
def smooth_data(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# 将 CSV 文件路径改为您的实际路径（使用原始字符串，避免转义问题）
data = pd.read_csv(r"F:\PERL\Perl_BaseLine\Perl_IDM\1113NN_lookup_table_NN50Epoch_data_IDM2.csv")
# data = pd.read_csv(r"F:\PERL\Perl_BaseLine\Perl_IDM\1106NN_lookup_table_PERL_data_IDM2.csv")
# data = pd.read_csv(r"F:\PERL\Perl_BaseLine\Perl_IDM\1113NN_lookup_table_Physical_data_IDM2.csv")


# 应用平滑处理，设置窗口大小
moderate_window_size = 3
filtered_desired_position = smooth_data(data['Desired_Location'], window_size=moderate_window_size)
filtered_rob_1_position = smooth_data(data['Rob_1_Location'], window_size=moderate_window_size)

# 生成与平滑数据匹配的 x 轴值
x_values_filtered = range(len(filtered_desired_position))

# 创建主图
fig, ax_main = plt.subplots(figsize=(14, 10))

# 绘制主图曲线
ax_main.plot(x_values_filtered, filtered_desired_position, label='Desired Position', linewidth=6, color='black', linestyle='--')
ax_main.plot(x_values_filtered, filtered_rob_1_position, label='Rob_2 Position', linewidth=6, color='blue')

# 设置标题和轴标签
ax_main.set_title('Online NN (Platoon)', fontsize=36)
# ax_main.set_title('Online PERL (Platoon)', fontsize=36)
# ax_main.set_title('Fixed Physical Model (Platoon)', fontsize=36)
ax_main.set_xlabel('Time Step', fontsize=32)
ax_main.set_ylabel('Position (m)', fontsize=32)
ax_main.tick_params(axis='both', labelsize=28)
ax_main.legend(fontsize=28)
ax_main.grid(True)

# 添加子图（嵌入局部放大图）
ax_inset = fig.add_axes([0.62, 0.23, 0.26, 0.26]) # [x, y, width, height] 设置子图位置
ax_inset.plot(x_values_filtered, filtered_desired_position, linewidth=6, color='black', linestyle='--')
ax_inset.plot(x_values_filtered, filtered_rob_1_position, linewidth=6, color='blue')

# 设置子图显示范围
ax_inset.set_xlim(250, 300)
ax_inset.set_ylim(24, 29,)
ax_inset.set_xlabel('Time Step', fontsize=20)
ax_inset.set_ylabel('Position (m)', fontsize=20)
ax_inset.tick_params(axis='both', labelsize=18)
ax_inset.grid(True)

# Adjust x-ticks to make them less dense
x_ticks = list(range(250, 300, 20))
ax_inset.set_xticks(x_ticks)
y_ticks = list(range(24, 29, 2))
ax_inset.set_yticks(y_ticks)

# 保存图像到指定文件夹
output_folder = r"F:\PERL\Perl_BaseLine\Fig"
output_filename = "Fig_p1_IDM2.png"
plt.savefig(f"{output_folder}\\{output_filename}", dpi=300, bbox_inches='tight')

# 显示图像
plt.show()
