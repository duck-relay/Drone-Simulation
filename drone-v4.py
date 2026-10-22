import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import heapq

# ==========================================
# CORE LOGIC (EKF & Sensors)
# ==========================================
class ConstraintEKF:
    def __init__(self):
        self.state = np.zeros((6, 1), dtype=np.float32)
        self.dt = 0.01  
        self.A = np.eye(6, dtype=np.float32)
        for i in range(3): self.A[i, i+3] = self.dt
        self.H = np.zeros((3, 6), dtype=np.float32)
        self.H[0, 0] = 1; self.H[1, 1] = 1; self.H[2, 2] = 1
        self.P = np.eye(6, dtype=np.float32) * 0.1
        self.Q = np.eye(6, dtype=np.float32) * 0.05
        self.R = np.eye(3, dtype=np.float32) * 2.0

    def apply_physical_constraints(self):
        if self.state[0, 0] <= 0.0:
            self.state[0, 0] = 0.0
            if self.state[3, 0] < 0: self.state[3, 0] = 0.0
        elif self.state[0, 0] >= 100.0:
            self.state[0, 0] = 100.0
            if self.state[3, 0] > 0: self.state[3, 0] = 0.0
        if self.state[1, 0] <= 0.0:
            self.state[1, 0] = 0.0
            if self.state[4, 0] < 0: self.state[4, 0] = 0.0
        elif self.state[1, 0] >= 100.0:
            self.state[1, 0] = 100.0
            if self.state[4, 0] > 0: self.state[4, 0] = 0.0
        if self.state[2, 0] < 0.0:
            self.state[2, 0] = 0.0
            if self.state[5, 0] < 0: self.state[5, 0] = 0.0

    def predict_imu(self, accel_world):
        B = np.zeros((6, 3), dtype=np.float32)
        dt_sq = 0.5 * self.dt**2
        for i in range(3):
            B[i, i] = dt_sq; B[i+3, i] = self.dt
        u = np.array(accel_world, dtype=np.float32).reshape(3, 1)
        self.state = self.A @ self.state + B @ u
        self.P = self.A @ self.P @ self.A.T + self.Q
        self.apply_physical_constraints()

    def update_visual(self, z_visual):
        y = z_visual - (self.H @ self.state)
        S = self.H @ self.P @ self.H.T + self.R
        K_gain = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K_gain @ y
        self.P = (np.eye(6, dtype=np.float32) - K_gain @ self.H) @ self.P
        self.apply_physical_constraints()

class SingleSensorKF:
    def __init__(self, process_noise=0.01, measurement_noise=0.5):
        self.z = 0.0; self.P = 1.0; self.Q = process_noise; self.R = measurement_noise
    def update(self, measurement):
        self.P = self.P + self.Q
        K = self.P / (self.P + self.R)
        self.z = self.z + K * (measurement - self.z)
        self.P = (1 - K) * self.P
        return self.z

class AltitudeFusionKF:
    def __init__(self):
        self.x = np.zeros((1, 1)); self.P = np.eye(1) * 1.0; self.Q = np.eye(1) * 0.05
        self.H = np.array([[1.0], [1.0]]) 
        self.R = np.array([[0.2, 0.0], [0.0, 1.0]])   
    def update(self, z_light, z_baro):
        self.P = self.P + self.Q
        z_meas = np.array([[z_light], [z_baro]])
        y = z_meas - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(1) - K @ self.H) @ self.P
        return self.x[0, 0]

def estimate_2_pnp(true_pos, fused_z):
    noise = np.random.normal(0, 0.8, 2)
    return np.array([true_pos[0] + noise[0], true_pos[1] + noise[1], fused_z]).reshape(3, 1)

def estimate_3_circles(true_pos, current_imu_est, L1, L2, fused_z):
    r1 = np.hypot(true_pos[0] - L1[0], true_pos[1] - L1[1]) + np.random.normal(0, 0.5)
    r2 = np.hypot(true_pos[0] - L2[0], true_pos[1] - L2[1]) + np.random.normal(0, 0.5)
    x1, y1 = L1[0], L1[1]; x2, y2 = L2[0], L2[1]
    d = np.hypot(x2 - x1, y2 - y1)
    if d > r1 + r2 or d < abs(r1 - r2) or d == 0:
        return np.array([true_pos[0] + np.random.normal(0, 1.0), true_pos[1] + np.random.normal(0, 1.0), fused_z]).reshape(3, 1)
    a = (r1**2 - r2**2 + d**2) / (2 * d)
    h = math.sqrt(abs(r1**2 - a**2))
    x3 = x1 + a * (x2 - x1) / d; y3 = y1 + a * (y2 - y1) / d
    root_a = np.array([x3 + h * (y2 - y1) / d, y3 - h * (x2 - x1) / d])
    root_b = np.array([x3 - h * (y2 - y1) / d, y3 + h * (x2 - x1) / d])
    imu_xy = np.array([current_imu_est[0, 0], current_imu_est[1, 0]])
    best_root = root_a if np.linalg.norm(root_a - imu_xy) < np.linalg.norm(root_b - imu_xy) else root_b
    return np.array([best_root[0], best_root[1], fused_z]).reshape(3, 1)

# ==========================================
# LINE OF SIGHT & PATHFINDING
# ==========================================
def line_intersects_aabb(p1, p2, aabb):
    """ Checks if camera LoS intersects a cuboid obstacle """
    min_x, max_x, min_y, max_y = aabb
    if (min_x <= p1[0] <= max_x and min_y <= p1[1] <= max_y) or \
       (min_x <= p2[0] <= max_x and min_y <= p2[1] <= max_y):
        return True
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    if dx == 0 and dy == 0: return False
    t0, t1 = 0.0, 1.0
    p = [-dx, dx, -dy, dy]
    q = [p1[0] - min_x, max_x - p1[0], p1[1] - min_y, max_y - p1[1]]
    for i in range(4):
        if p[i] == 0:
            if q[i] < 0: return False
        else:
            t = q[i] / p[i]
            if p[i] < 0:
                if t > t1: return False
                if t > t0: t0 = t
            else:
                if t < t0: return False
                if t < t1: t1 = t
    return t0 <= t1

def plan_astar_path(start, end, grid_size=100, obstacles=[]):
    grid = np.zeros((grid_size+1, grid_size+1), dtype=int)
    margin = 4.0 
    for obs_x, obs_y, obs_w, obs_d in obstacles:
        x_min = max(0, int(math.floor(obs_x - obs_w/2 - margin)))
        x_max = min(grid_size, int(math.ceil(obs_x + obs_w/2 + margin)))
        y_min = max(0, int(math.floor(obs_y - obs_d/2 - margin)))
        y_max = min(grid_size, int(math.ceil(obs_y + obs_d/2 + margin)))
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                grid[x, y] = 1

    start_int = (int(start[0]), int(start[1]))
    end_int = (int(end[0]), int(end[1]))
    grid[end_int[0], end_int[1]] = 0 
    grid[start_int[0], start_int[1]] = 0

    open_set = []
    heapq.heappush(open_set, (0, start_int))
    came_from = {}
    g_score = {start_int: 0}
    directions = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == end_int:
            path = []
            while current in came_from:
                path.append((float(current[0]), float(current[1])))
                current = came_from[current]
            path.reverse()
            if len(path) > 5:
                sampled_path = path[::5]
                if sampled_path[-1] != path[-1]: sampled_path.append(path[-1])
                return sampled_path
            return path

        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] <= grid_size and 0 <= neighbor[1] <= grid_size:
                if grid[neighbor[0], neighbor[1]] == 1: continue
                tentative_g = g_score[current] + math.hypot(dx, dy)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + math.hypot(neighbor[0]-end_int[0], neighbor[1]-end_int[1])
                    heapq.heappush(open_set, (f_score, neighbor))
    return []

# ==========================================
# PRE-COMPUTATION & VISUALIZATION
# ==========================================
if __name__ == "__main__":
    ekf = ConstraintEKF()
    kf_light = SingleSensorKF(process_noise=0.01, measurement_noise=0.3)
    kf_altitude_fusion = AltitudeFusionKF()

    # 1. FIRST: Generate exactly 2 random No-Fly Zones
    num_obstacles = 2
    obstacles = []
    for _ in range(num_obstacles):
        obs_x = np.random.uniform(20.0, 80.0)
        obs_y = np.random.uniform(20.0, 80.0)
        obs_w = np.random.uniform(10.0, 20.0) 
        obs_d = np.random.uniform(10.0, 20.0) 
        obstacles.append((obs_x, obs_y, obs_w, obs_d))

    # 2. SECOND: Generate Visual Landmarks avoiding obstacles
    num_landmarks = 20
    landmarks = []
    while len(landmarks) < num_landmarks:
        rand_x = np.random.uniform(5.0, 95.0)
        rand_y = np.random.uniform(5.0, 95.0)
        
        inside_obstacle = False
        margin = 1.0
        for obs_x, obs_y, obs_w, obs_d in obstacles:
            left_bound = obs_x - (obs_w / 2) - margin
            right_bound = obs_x + (obs_w / 2) + margin
            bottom_bound = obs_y - (obs_d / 2) - margin
            top_bound = obs_y + (obs_d / 2) + margin
            if (left_bound <= rand_x <= right_bound) and (bottom_bound <= rand_y <= top_bound):
                inside_obstacle = True
                break
                
        if not inside_obstacle:
            landmarks.append(np.array([rand_x, rand_y, 0.0]))

    # 3. Plan A* Flight Path
    start_pos = (0, 0)
    end_pos = (100, 100)
    path_2d = plan_astar_path(start_pos, end_pos, grid_size=100, obstacles=obstacles)

    waypoints = [np.array([0.0, 0.0, 25.0])] 
    for p in path_2d:
        waypoints.append(np.array([p[0], p[1], 25.0]))
    waypoints.append(np.array([100.0, 100.0, 0.0])) 

    current_wp = 0
    max_flight_ticks = 200000
    landed_successfully = False
    
    hist_x, hist_y, hist_z = [], [], []
    hist_l1, hist_l2 = [], [] 
    active_L1, active_L2 = landmarks[0], landmarks[1]

    # --- PRINT MANIFEST ---
    print("=" * 95)
    print("--- MISSION INITIALIZATION: FLIGHT MANIFEST ---")
    print(f"Generating {num_landmarks} completely random ground landmarks...")
    print("-" * 40)
    for i, lm in enumerate(landmarks):
        print(f"  Landmark {i:<2}: X={lm[0]:<7.2f} | Y={lm[1]:<7.2f}")
    print("-" * 40)
    print("Drone will trace this chaotic path and dynamically use the")
    print("CLOSEST 2 landmarks for geometric visual triangulation.")
    print("=" * 95)
    print(f"{'Tick':<6} | {'Status':<16} | {'Drone X':<10} | {'Drone Y':<10} | {'Alt (Z)':<10} | {'Target X':<10} | {'Target Y':<10}")
    print("-" * 95)
    
    tick = 0
    while current_wp < len(waypoints) and tick < max_flight_ticks:
        target = waypoints[current_wp]
        current_pos = ekf.state[0:3, 0]

        # 1. Check Line-of-Sight Occlusion
        occluded_in_2d = False
        drone_pos_2d = (ekf.state[0, 0], ekf.state[1, 0])
        l1_pos_2d = (active_L1[0], active_L1[1])
        l2_pos_2d = (active_L2[0], active_L2[1])

        for obs_x, obs_y, obs_w, obs_d in obstacles:
            margin = 1.0 
            aabb = (obs_x - obs_w/2 - margin, obs_x + obs_w/2 + margin, 
                    obs_y - obs_d/2 - margin, obs_y + obs_d/2 + margin)
            if line_intersects_aabb(drone_pos_2d, l1_pos_2d, aabb) or \
               line_intersects_aabb(drone_pos_2d, l2_pos_2d, aabb):
                occluded_in_2d = True
                break

        # 2. Check if flying directly OVER an obstacle (LiDAR Roof Collision)
        terrain_z = 0.0
        for obs_x, obs_y, obs_w, obs_d in obstacles:
            if (obs_x - obs_w/2) <= ekf.state[0,0] <= (obs_x + obs_w/2) and \
               (obs_y - obs_d/2) <= ekf.state[1,0] <= (obs_y + obs_d/2):
                terrain_z = 25.0
                break

        # Dynamic Altitude Climbing Logic
        current_target = target.copy()
        if occluded_in_2d:
            current_target[2] = 35.0  # Climb to clear view
            if ekf.state[2, 0] < 33.0:
                # Halt horizontal progress to wait and climb blind
                current_target[0] = ekf.state[0, 0]
                current_target[1] = ekf.state[1, 0]
                status_prefix = "Climbing (Blind)" 
            else:
                status_prefix = "High Alt"
        else:
            if current_wp == 0: status_prefix = "Takeoff"
            elif current_wp >= len(waypoints) - 2: status_prefix = "Landing"
            else: status_prefix = f"Tracing WP{current_wp}"

        status = f"{status_prefix}" if occluded_in_2d else f"{status_prefix} (Vis)"

        # Waypoint Acceptance
        acceptance_radius = 5.0 if current_wp < len(waypoints) - 1 else 1.5
        if np.linalg.norm(target - current_pos) < acceptance_radius:
            current_wp += 1
            if current_wp >= len(waypoints):
                ekf.state[0:3, 0] = target
                print(f"{tick:<6} | {'TOUCHDOWN':<16} | {ekf.state[0,0]:<10.3f} | {ekf.state[1,0]:<10.3f} | {ekf.state[2,0]:<10.3f} | {target[0]:<10.3f} | {target[1]:<10.3f}")
                landed_successfully = True
                break
            continue

        error = current_target - current_pos
        desired_vel = np.array([0.4, 0.4, 0.6]) * error
        desired_vel = np.clip(desired_vel, [-3.0, -3.0, -1.0], [3.0, 3.0, 1.0])
        accel_cmd = np.array([1.0, 1.0, 1.5]) * (desired_vel - ekf.state[3:6, 0])
        
        if np.linalg.norm(accel_cmd) > 2.0: 
            accel_cmd = (accel_cmd / np.linalg.norm(accel_cmd)) * 2.0

        noisy_accel = accel_cmd + np.random.normal(0, 0.2, 3)
        ekf.predict_imu(noisy_accel)

        # 3. Physically accurate LiDAR/Barometer Fusion
        true_z = ekf.state[2, 0] 
        measured_lidar_distance = (true_z - terrain_z) + np.random.normal(0, 0.4)
        z_baro_filtered = true_z + np.random.normal(0, 1.2) 

        if terrain_z > 0.0:
            exact_z = kf_altitude_fusion.update(z_baro_filtered, z_baro_filtered) 
        else:
            z_light_filtered = kf_light.update(measured_lidar_distance)
            exact_z = kf_altitude_fusion.update(z_light_filtered, z_baro_filtered)

        # 4. Stop X-Ray Vision! Only update EKF visually if NOT occluded.
        if tick % 10 == 0 and not occluded_in_2d:
            true_pos = ekf.state[0:3, 0].copy()
            imu_est = ekf.state[0:3, 0].copy().reshape(3, 1)
            
            distances = [(np.linalg.norm(true_pos[:2] - lm[:2]), lm) for lm in landmarks]
            distances.sort(key=lambda x: x[0])
            
            active_L1 = distances[0][1]
            active_L2 = distances[1][1]

            est_pnp = estimate_2_pnp(true_pos, exact_z)
            est_circles = estimate_3_circles(true_pos, imu_est, active_L1, active_L2, exact_z)
            ekf.update_visual((est_pnp + est_circles) / 2.0)

        # Output Telemetry 
        if tick % 1000 == 0:
             print(f"{tick:<6} | {status:<16} | {ekf.state[0,0]:<10.3f} | {ekf.state[1,0]:<10.3f} | {ekf.state[2,0]:<10.3f} | {target[0]:<10.3f} | {target[1]:<10.3f}")

        if tick % 10 == 0:
            hist_x.append(ekf.state[0, 0])
            hist_y.append(ekf.state[1, 0])
            hist_z.append(ekf.state[2, 0])
            
            hist_l1.append(active_L1 if not occluded_in_2d else np.array([ekf.state[0,0], ekf.state[1,0], ekf.state[2,0]]))
            hist_l2.append(active_L2 if not occluded_in_2d else np.array([ekf.state[0,0], ekf.state[1,0], ekf.state[2,0]]))

        tick += 1

    print("-" * 95)
    if landed_successfully:
        print(f"LANDED SAFELY AT: X={ekf.state[0,0]:.2f}, Y={ekf.state[1,0]:.2f}, Z={ekf.state[2,0]:.2f}")
    else:
        print(f"FLIGHT TIMEOUT: Struck at position X={ekf.state[0,0]:.2f}, Y={ekf.state[1,0]:.2f}, Z={ekf.state[2,0]:.2f}")

    # ==========================================
    # MATPLOTLIB 3D ANIMATION ENGINE
    # ==========================================
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_xlim([0, 100]); ax.set_ylim([0, 100]); ax.set_zlim([0, 40])
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Altitude (m)')
    ax.set_title('A* Navigation with Dynamic Altitude Climbing')

    for obs_x, obs_y, obs_w, obs_d in obstacles:
        bottom_left_x = obs_x - (obs_w / 2)
        bottom_left_y = obs_y - (obs_d / 2)
        z_base = 0
        z_height = 25 
        ax.bar3d(bottom_left_x, bottom_left_y, z_base, obs_w, obs_d, z_height, color='red', alpha=0.3, edgecolor='darkred')

    lx = [lm[0] for lm in landmarks]
    ly = [lm[1] for lm in landmarks]
    lz = [lm[2] for lm in landmarks]
    ax.scatter(lx, ly, lz, c='grey', marker='s', s=40, label='Visual Landmarks')

    px = [wp[0] for wp in waypoints]
    py = [wp[1] for wp in waypoints]
    pz = [wp[2] for wp in waypoints]
    ax.plot(px, py, pz, 'y--', linewidth=2, label='A* Planned Route')

    drone_dot, = ax.plot([], [], [], 'bo', markersize=8, label='Drone')
    trail_line, = ax.plot([], [], [], 'dodgerblue', alpha=0.4, linewidth=2)
    laser1, = ax.plot([], [], [], 'lime', linestyle='--', linewidth=1.5, label='Camera Lock')
    laser2, = ax.plot([], [], [], 'lime', linestyle='--', linewidth=1.5)
    
    ax.legend()
    step_size = 5 
    
    def update_graph(num):
        idx = num * step_size
        if idx >= len(hist_x):
            idx = len(hist_x) - 1
            
        dx, dy, dz = hist_x[idx], hist_y[idx], hist_z[idx]
        
        drone_dot.set_data([dx], [dy])
        drone_dot.set_3d_properties([dz])
        
        tail_start = max(0, idx - 150)
        trail_line.set_data(hist_x[tail_start:idx], hist_y[tail_start:idx])
        trail_line.set_3d_properties(hist_z[tail_start:idx])
        
        target1 = hist_l1[idx]
        target2 = hist_l2[idx]
        
        laser1.set_data([dx, target1[0]], [dy, target1[1]])
        laser1.set_3d_properties([dz, target1[2]])
        
        laser2.set_data([dx, target2[0]], [dy, target2[1]])
        laser2.set_3d_properties([dz, target2[2]])
        
        return drone_dot, trail_line, laser1, laser2

    total_frames = len(hist_x) // step_size
    ani = animation.FuncAnimation(
        fig, update_graph, frames=total_frames, 
        interval=20, blit=False, repeat=False
    )

    plt.show()