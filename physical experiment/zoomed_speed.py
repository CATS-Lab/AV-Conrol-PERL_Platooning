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
# data = pd.read_csv(r"F:\PERL\Perl_BaseLine\1031NN_lookup_table_NN50Epoch_data.csv")
# data = pd.read_csv(r"F:\PERL\Perl_BaseLine\1031NN_lookup_table_PERL_data.csv")
data = pd.read_csv(r"F:\PERL\Perl_BaseLine\1031NN_lookup_table_PhysicalModel_data.csv")

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
ax_main.plot(x_values_filtered, filtered_rob_1_speed, label='Rob_1 Speed', linewidth=6, color='blue')
# ax_main.set_title('Online NN (Single Robot)', fontsize=36)
# ax_main.set_title('Online PERL (Single Robot)', fontsize=36)
ax_main.set_title('Fixed Physical Model (Single Robot)', fontsize=36)
ax_main.set_xlabel('Time Step', fontsize=32)
ax_main.set_ylabel('Speed (m/s)', fontsize=32)
ax_main.tick_params(axis='both', labelsize=28)
ax_main.legend(fontsize=28)
ax_main.grid(True)

# Adjusted zoomed-in subplot at the bottom-right of the main plot
ax_inset = fig.add_axes([0.61, 0.22, 0.27, 0.27])  # Adjusted [x, y, width, height] for better positioning
ax_inset.plot(x_values_filtered, filtered_desired_speed, linewidth=6, color='black', linestyle='--')
ax_inset.plot(x_values_filtered, filtered_rob_1_speed, linewidth=4, color='blue')
ax_inset.set_xlim(320, 380)
ax_inset.set_ylim(0.54, 0.62)
ax_inset.set_xlabel('Time Step', fontsize=20)  # Add x-axis label
ax_inset.set_ylabel('Speed (m/s)', fontsize=20)  # Add y-axis label
ax_inset.tick_params(axis='both', which='major', labelsize=18)

# Adjust x-ticks to make them less dense
x_ticks = list(range(320, 381, 20))  # Increase interval to 10
ax_inset.set_xticks(x_ticks)

ax_inset.grid(True)

# Save the figure to the specified folder
output_folder = r"F:\PERL\Perl_BaseLine\Fig_wmf"
output_filename = "Fig.4.png"
plt.savefig(f"{output_folder}\\{output_filename}", dpi=300, bbox_inches='tight')

plt.show()

