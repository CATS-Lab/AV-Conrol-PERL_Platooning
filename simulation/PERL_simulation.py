import numpy as np
from scipy.optimize import minimize
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

# ------------------------ Simulation Environment ------------------------
dt = 0.1  # Time interval between data points (in seconds)
T = 300  # Simulation period (30 seconds)
I = 4  # Number of vehicles in the platoon, excluding the lead vehicle
tau = np.array([0.15] * I)  # Vehicle delay (0.15 seconds)
np.random.seed(1)

# ------------------------ MPC Parameters ------------------------
N = 5  # Prediction horizon
w = [0.1, 1, 0.1, 0.1]  # Weights in the objective function

d_range = [5, 80]  # Minimum and maximum distance range
v_range = [5, 50]  # Minimum and maximum velocity range
a_range = [-5, 5]  # Minimum and maximum acceleration range

# ------------------------ Input Reference Trajectory ------------------------
# Reference trajectories include the lead vehicle and platoon vehicles
input_file_path = ['./data/IDM_trajectory_1.csv', './data/OpenACC_trajectory_1.csv']
ref_df = pd.read_csv(input_file_path[1])

# Initialize target state variables for position, velocity, and acceleration
p_tar = np.zeros((T, I + 1))
v_tar = np.zeros((T, I + 1))
a_tar = np.zeros((T, I + 1))

# Extract speed data for the reference trajectories
speed_columns = [f'Speed_{i}' for i in range(1, I + 2)]
speed_data = ref_df[speed_columns].values[:T]
v_tar[:, :] = speed_data

# Set initial positions for the vehicles in the platoon
p_tar[0, 0] = 80
p_tar[0, 1] = 60
p_tar[0, 2] = 40
p_tar[0, 3] = 20
p_tar[0, 4] = 0

# Compute positions and accelerations for the reference trajectories
for i in range(1, T):
    p_tar[i, :] = p_tar[i - 1, :] + (v_tar[i - 1, :] + v_tar[i, :]) / 2 * dt
for i in range(0, T - 1):
    a_tar[i, :] = (v_tar[i + 1, :] - v_tar[i, :]) / dt

# Combine position, velocity, and acceleration into a target state tensor
target_state = np.stack((p_tar, v_tar, a_tar), axis=0)

train_steps = 20  # Number of steps for model training

method = "N"  # Method identifier
trajectory_list = ["1", "2", "3", "4"]  # List of trajectory IDs


# ------------------------ algorithm ------------------------

# ------------------------ Algorithm Functions ------------------------
def thread_MPC(initial_state, last_u, step):
    """
    Solves the MPC optimization problem for a platoon of vehicles.
    """

    def objective(u, w, tar_state, last_u):
        """Objective function for MPC optimization."""
        u = np.reshape(u, (N, I))
        p, v, a = np.zeros_like(u), np.zeros_like(u), np.zeros_like(u)
        p[0, :], v[0, :], a[0, :] = initial_state

        # Compute predicted states over the prediction horizon
        for t in range(1, N):
            p[t] = p[t - 1] + v[t - 1] * dt + 0.5 * a[t - 1] * (dt ** 2)
            v[t] = v[t - 1] + a[t - 1] * dt
            a[t] = dt / tau * (u[t - 1] - v[t - 1])

        # Compute control input changes (du) and errors
        du = np.diff(np.vstack((np.reshape(last_u, (1, I)), u)), axis=0)
        errors = (p - tar_state[0, :, 1:]) ** 2, (v - tar_state[1, :, 1:]) ** 2, (a - tar_state[2, :, 1:]) ** 2
        obj = sum(np.sum(err) * weight for err, weight in zip(errors, w)) + np.sum(du) ** 2 * w[3]
        return obj

    def build_state(u, t, idx):
        """Builds the state trajectory for a specific vehicle."""
        p, v, a = np.zeros(t + 1), np.zeros(t + 1), np.zeros(t + 1)
        p[0], v[0], a[0] = initial_state[:, idx]
        u = np.reshape(u, (N, I))[:, idx]
        for _t in range(1, t + 1):
            p[_t] = p[_t - 1] + v[_t - 1] * dt + 0.5 * a[_t - 1] * (dt ** 2)
            v[_t] = v[_t - 1] + a[_t - 1] * dt
            a[_t] = dt / tau[idx] * (u[_t - 1] - v[_t - 1])
        return p, v, a

    def constraint_safety_dis(u, t, idx, low_b):
        """Safety distance constraint."""
        if idx == 0:
            p_l = target_state[0, step:step + t + 1, 0]
        else:
            p_l, _, _ = build_state(u, t, idx - 1)
        p_f, _, _ = build_state(u, t, idx)
        return p_l[t] - p_f[t] - low_b

    def constraint_platoon_dis(u, t, idx, up_b):
        """Platoon distance constraint."""
        if idx == 0:
            p_l = target_state[0, step:step + t + 1, 0]
        else:
            p_l, _, _ = build_state(u, t, idx - 1)
        p_f, _, _ = build_state(u, t, idx)
        return up_b - (p_l[t] - p_f[t])

    def constraint_speed_range_lower(u, t, idx, low_b):
        """Lower bound constraint for speed."""
        _, v, _ = build_state(u, t, idx)
        return v[t] - low_b

    def constraint_speed_range_upper(u, t, idx, up_b):
        """Upper bound constraint for speed."""
        _, v, _ = build_state(u, t, idx)
        return up_b - v[t]

    def constraint_acceleration_range_lower(u, t, idx, low_b):
        """Lower bound constraint for acceleration."""
        _, _, a = build_state(u, t, idx)
        return a[t] - low_b

    def constraint_acceleration_range_upper(u, t, idx, up_b):
        """Upper bound constraint for acceleration."""
        _, _, a = build_state(u, t, idx)
        return up_b - a[t]

    def setup_constraints(N, I, d_range, v_range, a_range):
        """Sets up constraints for the MPC problem."""
        cons = []
        for t in range(N):
            for i in range(I):
                cons.extend([
                    {'type': 'ineq', 'fun': lambda u, t_p=t, idx=i: constraint_safety_dis(u, t_p, idx, d_range[0])},
                    {'type': 'ineq', 'fun': lambda u, t_p=t, idx=i: constraint_platoon_dis(u, t_p, idx, d_range[1])},
                    {'type': 'ineq',
                     'fun': lambda u, t_p=t, idx=i: constraint_speed_range_lower(u, t_p, idx, v_range[0])},
                    {'type': 'ineq',
                     'fun': lambda u, t_p=t, idx=i: constraint_speed_range_upper(u, t_p, idx, v_range[1])},
                    {'type': 'ineq',
                     'fun': lambda u, t_p=t, idx=i: constraint_acceleration_range_lower(u, t_p, idx, a_range[0])},
                    {'type': 'ineq',
                     'fun': lambda u, t_p=t, idx=i: constraint_acceleration_range_upper(u, t_p, idx, a_range[1])},
                ])
        return cons

    u = np.tile(last_u, (N, 1)).flatten()

    # Setup constraints
    cons = setup_constraints(N, I, d_range, v_range, a_range)

    # Run the optimization
    solution = minimize(objective, u, args=(w, target_state[:, step:step + N], last_u),
                        method='SLSQP', constraints=cons, options={'maxiter': 100})
    if solution.success:
        print("Optimization succeeded.")
    else:
        print("Optimization failed:", solution.message)

    return solution.x[:I]


def thread_learning(speed_command, previous_speed, model):
    """
    Learns the relationship between speed command and vehicle response.
    """
    data = np.column_stack((speed_command, previous_speed))
    speed_residual = model.predict(data)
    speed_residual = np.reshape(speed_residual, (-1))
    RPM_command = (speed_residual + speed_command - 25) / 615.4
    return RPM_command


def disturb(current_u):
    """
    Introduces disturbance to the control input.
    """
    speed = current_u * 615.4 + 25
    return speed + (speed ** 2 * 0.01 - 3 + np.random.normal(0, 1, speed.shape))


def thread_simulator(current_u, current_state, step):
    """
    Simulates the vehicle dynamics for the given control input.
    """
    actual_u = disturb(current_u)
    current_state[0, step] = current_state[0, step - 1] + current_state[1, step - 1] * dt + current_state[
        2, step - 1] * (dt ** 2) / 2
    current_state[1, step] = current_state[1, step - 1] + current_state[2, step - 1] * dt
    current_state[2, step] = dt / tau * (actual_u - current_state[1, step - 1])
    return current_state, actual_u


def build_NN_model():
    """
    Builds and trains a neural network model for learning vehicle dynamics.
    """
    loss = tf.keras.losses.MeanSquaredError()
    trainer = tf.keras.optimizers.Adam(0.001, 0.9)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(2),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(4, activation="relu"),
        tf.keras.layers.Dense(1)
    ])
    model.compile(loss=loss, optimizer=trainer)

    MPC_inputs = np.arange(15, 35, 0.2)
    MPC_inputs = np.reshape(MPC_inputs, (-1, 1))
    previous_speed = np.arange(14, 34, 0.2)
    previous_speed = np.reshape(previous_speed, (-1, 1))
    X_train = np.hstack((MPC_inputs, previous_speed))
    RPM_command = (MPC_inputs - 25) / 615.4
    actual_inputs = disturb(RPM_command)
    Y_train = MPC_inputs - actual_inputs

    model.fit(X_train, Y_train, epochs=100, batch_size=20)

    return model


def online_learning():
    """
    Performs online learning to adapt the vehicle model.
    """
    model = build_NN_model()

    p = np.zeros((T, I))
    v = np.zeros((T, I))
    a = np.zeros((T, I))
    p[0] = p_tar[0, 1:]
    v[0] = v_tar[0, 1:]
    a[0] = a_tar[0, 1:]
    states = np.stack([p, v, a], axis=0)
    actual_input = a[0] / dt * tau + v[0]
    X_new_1 = np.zeros((train_steps, I))
    X_new_2 = np.zeros((train_steps, I))
    Y_new = np.zeros((train_steps, I))
    for t in range(T - N):
        MPC_desired_input = thread_MPC(states[:, t, :], actual_input, t)
        RL_desired_input = thread_learning(MPC_desired_input, states[1, t, :], model)
        states, actual_input = thread_simulator(RL_desired_input, states, t + 1)
        X_new_1[t % train_steps] = MPC_desired_input
        X_new_2[t % train_steps] = states[1, t, :]
        Y_new[t % train_steps] = MPC_desired_input - actual_input

        print(f"finish time {t}, MPC_u: {MPC_desired_input}, current_u: {RL_desired_input}, real_u: {actual_input}")
        if t % train_steps == train_steps - 1:
            X_train = np.hstack((np.reshape(X_new_1, (-1, 1)), np.reshape(X_new_2, (-1, 1))))
            Y_train = np.reshape(Y_new, (-1, 1))
            model.fit(X_train, Y_train, epochs=10, batch_size=10)
            X_new_1 = np.zeros((train_steps, I))
            X_new_2 = np.zeros((train_steps, I))
            Y_new = np.zeros((train_steps, I))
    np.save(f'./results/states-' + method + '-1.npy', states)


def analyze():
    """
    Analyzes the simulation results and outputs errors.
    """
    for traj_id in trajectory_list:
        states = np.load(f"./results/states-" + method + f"-{traj_id}.npy")
        p_error = np.abs((states[0, 2:T - N] - p_tar[2:T - N, 1:]))
        p_error_max = np.max(p_error)
        p_error_sum = np.sum(p_error)
        v_error = np.abs((states[1, 2:T - N] - v_tar[2:T - N, 1:]))
        v_error_max = np.max(v_error)
        v_error_sum = np.sum(v_error)
        df = pd.DataFrame(np.array([p_error_sum, v_error_sum, p_error_max, v_error_max]))
        df.to_csv(f'./results/output-' + traj_id + '.csv', index=False)


def draw_traj_figures():
    """
    Draws trajectory figures for the simulation results.
    """
    matplotlib.rcParams['font.family'] = 'Times New Roman'
    matplotlib.rcParams['font.size'] = 20

    colors = ['#FE5C5C', '#75ABDE', '#FE9C43', '#95C061', 'black']
    colors_ = ['#808080', '#7C5674', '#748A54', '#A6886D']
    for traj_id in trajectory_list:
        states = np.load(f"./results/states-" + method + f"-{traj_id}.npy")

        time = np.arange(0, 0.1 * (T - N), 0.1)

        plt.plot(time, p_tar[:T - N, 0], '-', color=colors[-1], label=f'vehicle {0}', linewidth=2)
        for i in range(I):
            plt.plot(time, p_tar[:T - N, i + 1], '-', color=colors[i], label=f'vehicle {i + 1} (R)',
                     linewidth=2)
            plt.plot(time, states[0, :T - N, i], ':', color=colors_[i], label=f'vehicle {i + 1} (A)', linewidth=2)

        plt.xlabel('Time (seconds)', fontsize=35)
        plt.ylabel('Position ($m$)', fontsize=35)
        plt.legend(loc='lower right', fontsize=13, framealpha=1)
        plt.grid(True)

        output_path = './results/traj-' + traj_id + '.svg'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', format='svg')
        plt.close()

    for traj_id in trajectory_list:
        states = np.load(f"./results/states-" + method + f"-{traj_id}.npy")

        time = np.arange(0, 0.1 * (T - N), 0.1)

        for i in range(I):
            plt.plot(time, states[1, :T - N, i] - v_tar[:T - N, i + 1], '-', color=colors[i],
                     label=f'vehicle {i + 1}', linewidth=1)

        plt.xlabel('Time (seconds)', fontsize=25)
        plt.ylabel('Velocity error ($m/s$)', fontsize=25)
        plt.legend(loc='upper right', fontsize=18)
        plt.grid(True)

        output_path = './results/velocity-' + traj_id + '.svg'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', format='svg')
        plt.close()


# ------------------------ Main Function ------------------------
if __name__ == "__main__":
    # Perform online learning
    online_learning()

    # Generate trajectory figures
    # draw_traj_figures()

    # Analyze results
    # analyze()