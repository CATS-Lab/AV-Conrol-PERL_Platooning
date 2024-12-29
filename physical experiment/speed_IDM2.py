import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置全局字体为 Times New Roman
rcParams['font.family'] = 'Times New Roman'

# Define moving average function
def smooth_data(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# 将 CSV 文件路径改为您的实际路径（使用原始字符串，避免转义问题）
# data = pd.read_csv(r"F:\PERL\Perl_BaseLine\Perl_IDM\1113NN_lookup_table_NN50Epoch_data_IDM2.csv")
# data = pd.read_csv(r"F:\PERL\Perl_BaseLine\Perl_IDM\1106NN_lookup_table_PERL_data_IDM2.csv")
data = pd.read_csv(r"F:\PERL\Perl_BaseLine\Perl_IDM\1113NN_lookup_table_Physical_data_IDM2.csv")


# Apply moderate smoothing with a window size of 3
moderate_window_size = 3
filtered_desired_speed = smooth_data(data['Desired_Speed'], window_size=moderate_window_size)
filtered_rob_1_speed = smooth_data(data['Rob_1_Speed'], window_size=moderate_window_size)

# Generate x-values to match the filtered data
x_values_filtered = range(len(filtered_desired_speed))

# Create the main plot
fig, ax_main = plt.subplots(figsize=(14, 10))

# Main plot
ax_main.plot(x_values_filtered, filtered_desired_speed, label='Desired Speed', linewidth=6, color='black', linestyle='--')
ax_main.plot(x_values_filtered, filtered_rob_1_speed, label='Rob_2 Speed', linewidth=6, color='blue')
# ax_main.set_title('Online NN (Platoon)', fontsize=36)
# ax_main.set_title('Online PERL (Platoon)', fontsize=36)
ax_main.set_title('Fixed Physical Model (Platoon)', fontsize=36)
ax_main.set_xlabel('Time Step', fontsize=32)
ax_main.set_ylabel('Speed (m/s)', fontsize=32)
ax_main.tick_params(axis='both', labelsize=28)
ax_main.legend(fontsize=28)
ax_main.grid(True)


# Save the figure to the specified folder
output_folder = r"F:\PERL\Perl_BaseLine\Fig"
output_filename = "Fig.4_IDM_2.png"
plt.savefig(f"{output_folder}\\{output_filename}", dpi=300, bbox_inches='tight')

plt.show()

