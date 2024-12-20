import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dt = 0.1  # Time step in seconds
T = 300  # Total number of simulation steps
trj_num = 5  # Number of trajectories

def OpenACC_trajectory(output_path):
    """
    Generate OpenACC trajectories by extracting and segmenting data from a CSV file.

    Args:
        output_path (str): Path to save the generated trajectory files.
    """
    # Read the CSV file
    csv_file_path = './data/ASta_platoon1.csv'
    df = pd.read_csv(csv_file_path, skiprows=5, header=0, low_memory=False)

    # Extract relevant columns
    columns_to_extract = ['Time', 'Speed1', 'Speed2', 'Speed3', 'Speed4', 'Speed5']
    df = df[columns_to_extract]
    df = df[200:].reset_index(drop=True)

    # Assign segment IDs based on index
    df['ID'] = (df.index // T) + 1

    # Save each segment to a separate CSV file
    for i in range(1, trj_num + 1):
        df_segment = df[df['ID'] == i].copy()
        df_segment.drop(columns=['ID'], inplace=True)

        # Adjust Time column to start from 0
        df_segment['Time'] = df_segment['Time'] - df_segment['Time'].iloc[0]
        df_segment = df_segment[df_segment['Time'] < T * dt]
        df_segment.to_csv(f"{output_path}/OpenACC_trajectory_{i}.csv", index=False)

def IDM_trajectory(output_path):
    """
    Generate IDM (Intelligent Driver Model) trajectories.

    Args:
        output_path (str): Path to save the generated trajectory files.
    """
    car_num = 5  # Number of vehicles

    # IDM parameters
    v0 = 33.3  # Maximum speed (m/s)
    T0 = 2.8  # Time headway (s)
    s0 = 2  # Minimum distance (m)
    amax = 0.73  # Maximum acceleration (m/s^2)
    b = 1.67  # Comfortable deceleration (m/s^2)

    def IDM(delta_d, v, delta_v):
        """Calculate IDM acceleration."""
        s_star = s0 + max(0, v * T0 + (v * delta_v) / (2 * ((amax * b) ** 0.5)))
        small_value = 1e-5  # To avoid division by zero
        return amax * (1 - (v / v0) ** 4 - (s_star / (delta_d + small_value)) ** 2)

    # Initialize positions and velocities for all vehicles
    p = np.zeros((car_num, T + 1))
    v = np.zeros((car_num, T + 1))

    # Set initial conditions for the lead vehicle
    p[0, 0] = 0
    v[0, 0] = 20

    # Generate trajectory for the lead vehicle
    for t in range(1, T + 1):
        v[0, t] = 20 + 5 * np.sin(0.02 * t)
        p[0, t] = p[0, t - 1] + dt * (v[0, t - 1] + v[0, t]) / 2

    # Generate trajectories for following vehicles
    for i in range(1, car_num):
        p[i, 0] = p[i - 1, 0] - (s0 + T0 * v[i - 1, 0])  # Start behind the lead vehicle
        v[i, 0] = 20

    for t in range(T):
        for i in range(1, car_num):
            delta_d = p[i - 1, t] - p[i, t]
            delta_v = v[i, t] - v[i - 1, t]
            a = IDM(delta_d, v[i, t], delta_v)
            v[i, t + 1] = v[i, t] + dt * a
            p[i, t + 1] = p[i, t] + dt * (v[i, t] + v[i, t + 1]) / 2

    # Save trajectories to a CSV file
    time_array = np.arange(0, (T + 1) * dt, dt)
    data = {'Time': time_array}
    for j in range(5):
        data[f'Speed_{j + 1}'] = np.round(v[j], 2)
    df = pd.DataFrame(data)
    df.to_csv(f"{output_path}/IDM_trajectory_4.csv", index=False)

def draw_trajectory(file_path, output_path):
    """
    Plot trajectories from a CSV file.

    Args:
        file_path (str): Path to the CSV file containing trajectories.
        output_path (str): Path to save the plotted figure.
    """
    # Read the CSV file
    data = pd.read_csv(file_path)

    # Plot the trajectories
    plt.figure(figsize=(12, 8))

    # Plot speed for each vehicle
    for i, column in enumerate(data.columns[1:], start=1):
        plt.plot(data['Time'], data[column], label=f"Vehicle {i}", linewidth=4)

    # Set labels and styles
    plt.xlabel('Time (s)', fontsize=40, fontname='Times New Roman')
    plt.ylabel('Speed (m/s)', fontsize=40, fontname='Times New Roman')
    plt.legend(prop={'size': 30, 'family': 'Times New Roman'}, loc='lower right')
    plt.xticks(fontsize=40, fontname='Times New Roman')
    plt.yticks(fontsize=40, fontname='Times New Roman')

    # Adjust layout and save as SVG
    plt.tight_layout()
    plt.savefig(output_path, format='svg')

if __name__ == "__main__":
    # Generate trajectories and optionally plot them
    OpenACC_trajectory('./data')
    IDM_trajectory('./data')
    # draw_trajectory('./data/OpenACC_trajectory_1.csv', './results/OpenACC_1.svg')
    # draw_trajectory('./data/IDM_trajectory_1.csv', './results/IDM_1.svg')