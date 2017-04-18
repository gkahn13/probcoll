
import numpy as np
from robots.bebop2d.ros.utils import pose_to_posquat, posquat_to_pose
import robots.bebop2d.ros.transformations as tft
import math
from robots.bebop2d.ros.transformations import quaternion_matrix
def angular_vel_from_radius(radius,linear_velocity = 2.0):
    return float(linear_velocity)/radius

def circular_path_state(t, height,T, dt = 0.05, angular_velocity = 0.333333):
    desired_state = np.zeros([T,3])
    desired_yaw = np.zeros([T,1])
    for i in xrange(T):
        theta = float(t + i)*dt*angular_velocity
        desired_state[i][0] = 6*math.sin(theta)
        desired_state[i][1] = 6 -6*math.cos(theta)
        desired_state[i][2] = height
        desired_yaw[i] = theta
    return desired_state, desired_yaw

def eight_shape_path(t, height, T, angular_velocity = 0.333333):
    desired_state = np.zeros([T,3])
    for i in xrange(T):
        theta = float(t + i)/T*angular_velocity
        if theta%(4*math.pi)  < 2*math.pi:
            desired_state[i][1] = 6-6*math.cos(theta)
        else:
            desired_state[i][1] = -6+6*math.cos(theta)
        desired_state[i][0] = 6*math.sin(theta)
        desired_state[i][2] = height
    return desired_state

def circular_path(t):
    return 6*math.sin(t),-6+6*math.cos(t)
def test_func_path(t):
    return t, 5*math.sin(t)+4*math.cos(2*t)-4
def test_func_path1(t):
    return 4*math.sin(t), 5*math.sin(t)+4*math.cos(2*t)-4   
def test_func_path2(t):
    return 4*math.sin(t)+10*math.cos(3*t)-10, 5*math.sin(t)+4*math.cos(2*t)-4
def test_func_path3(t):
    if t < 2:
        return 0,-2*t
    elif t <3:
        return 2*(t-2),-4
    elif t <5:
        return 2, -4 + 2*(t-3)
    else: return 2+abs(math.sin(math.pi*(t-5))), -2*(t-5)
def test_func_path4(t):
    if t < 1: 
        return 2*t,0
    elif t < 2:
        return 2-2*(t-1), -2*(t-1)
    elif t <3:
        return 2*(t-2), -2
    elif t <4:
        return 2 + 2*(t-3), -2 - 2*(t-3)
    elif t <5:
        return 2 + 2*(t-3), -4 + 2*(t-4)
    elif t < 6:
        return 6 - 2*(t-5), -2 - 2*(t-5)
    elif t <9:
        return 4, -4 -2*(t-6)
    elif t < 10:
        return 4, -10 + 2*(t-9)
    elif t < 11:
        return 4 + 2*(t-10), -8
    elif t < 12:
        return 6 - 2*(t-11), -8
    elif t < 13:
        return 4, -8 + 2*(t-12)
    else:
        return 4 + 2*(t-13), -6
def test_func_path5(t):
    if t < 1:
        return 0, 3*t
    elif t <3:
        return 1-math.cos(math.pi*(t-1)), 3+math.sin(math.pi*(t-1))
    elif t < 4:
        return 2*(t-3), 3-3*(t-3)
    elif t <6:
        return 2, 2*(t-4)
    elif t < 8:
        return 2, 4-2*(t-6)
    elif t < 9:
        return 2 + 2*(t-8), 0
    elif t <11:
        return 4, 2*(t-9)
    elif t < 13:
        return 4, 4-2*(t-11)
    else:
        return 4 + 2*(t-13), 0

def find_position(path_func, t, dt):
    x_t, y_t = path_func(t)
    x_tplus, y_tplus = path_func(t+dt)
    deltaT = 2.0/(np.linalg.norm([x_tplus-x_t, y_tplus - y_t])/dt)*dt
    return path_func(t+deltaT)
def general_path(path_func, t, height, T, dt = 0.05):
    #func should be a function taking t as input while retuning x_t and y_t
    desired_state = np.zeros([T,3])
    for i in xrange(T):
        cur_t = float(t + i) * dt
        desired_state[i][0], desired_state[i][1] = find_position(path_func, cur_t, dt)
        desired_state[i][2] = height
    return desired_state
def generate_orientation(theta):
    m = np.zeros([4,4])
    m[0][0] = math.cos(theta)    
    m[0][1] = -math.sin(theta)  
    m[1][0] = math.sin(theta)  
    m[1][1] = math.cos(theta)
    m[2][2] = 1
    p, q_wxyz = pose_to_posquat(m)
    if q_wxyz[0] <0:
        q_wxyz = q_wxyz*(-1)
    # print q_wxyz
##        import IPython; IPython.embed()
    return q_wxyz

def velocity_from_path(path, dt = 0.05):
    # assuming each adjacent path element pair differs by dts
    desired_vel = np.zeros(path.shape)
    desired_vel[:-1] = np.subtract(path[1:],path[:-1])/dt
    desired_vel[-1] = desired_vel[-2]
    return desired_vel

def circular_path_orientation(t,T, angular_velocity = 0.333333):
    desired_state = np.zeros([T,4])
    for i in xrange(T):
        theta = float(t + i)/T*angular_velocity
        
        desired_state[i] = generate_orientation(theta)
    return desired_state
def direction_vector_from_velocity(velocity):
    vel_norm  = np.linalg.norm(velocity[:-1])
    m = np.zeros([4,4])
    m[0][0] = float(velocity[0])/vel_norm
    m[0][1] = -float(velocity[1])/vel_norm
    m[1][1] = float(velocity[0])/vel_norm
    m[1][0] = float(velocity[1])/vel_norm
    m[2][2] = 1
    return m
def orientation_from_single_velocity(t, T, velocity):
    vel_matrix = direction_vector_from_velocity(velocity)
    p, q_wxyz = pose_to_posquat(vel_matrix)
    if q_wxyz[0] <0:
        q_wxyz = q_wxyz*(-1)
#        import IPython; IPython.embed()
    # print q_wxyz
    return q_wxyz
def orientation_from_velocity(t,T,velocity):
    if len(np.asarray(velocity).shape) ==1:
        return orientation_from_single_velocity(t, T, velocity)
    else:
        desired_ori = np.zeros([T,4])
        for i in xrange(T):
            desired_ori[i] = orientation_from_single_velocity(t, T, velocity[i])
        return desired_ori

def cost_temp(x, target, weight_pos =2, weight_yaw = 0):

    cost = weight_pos*((x[0]-target[0])**2 + (x[1]-target[1])**2 + (x[2]-target[2])**2)
    # cost += weight_yaw*(x[3]-target[3])**2
    return cost
def cost_speed(u, v_target, weight_speed = 2):
    return weight_speed*((np.linalg.norm(v[:-1])-v_target)**2 + v[-1]**2)

def cost_sh(x, target, weight_height = 2):

    return weight_height* (x[2] - target)**2

def cost_con(x, target, weight_pos =2, weight_yaw = 2):

    cost = weight_pos*((x[0]-target[0])**2 + (x[1]-target[1])**2 + (x[2]-target[2])**2)
    cost += weight_yaw*(x[3]-target[3])**2
    return cost

def cost_vary_cont(u_t, vel_t, weight = 10):
    return weight*((u_t[0] - vel_t[0])**2 + (u_t[1] - vel_t[1])**2 + (u_t[2] - vel_t[2])**2)

def gradient_temp(func, x_0, target, epsilon = 1e-5):
    gradient = np.zeros([len(x_0)])
    for i in xrange(len(x_0)):
        x_0[i] += epsilon
        f1 = func(x_0, target)
        x_0[i] -= 2*epsilon
        f2 = func(x_0, target)
        x_0[i] += epsilon
        gradient[i] = (f1-f2)/(2*epsilon)
    return gradient

def hessian_temp(func, x_0, target, epsilon = 1e-5):
    hes = np.zeros([len(x_0),len(x_0)])
    for i in xrange(len(x_0)):
        # import IPython; IPython.embed()
        x_0[i] += epsilon
        f1 = gradient_temp(func,x_0, target)
        x_0[i] -= 2*epsilon
        f2 = gradient_temp(func,x_0, target)
        x_0[i] += epsilon
        hes[i,:] = (f1-f2)/(2*epsilon)
    return hes
def cost_smooth(u_t, u_y,  weight_control = 10):
    # diff = abs(u_t[3]%(2*math.pi) - u_y[3]%(2*math.pi))
    # diff = min(diff, 2*math.pi- diff)
    return weight_control*((u_t[0] - u_y[0])**2+(u_t[1] - u_y[1])**2+(u_t[2] - u_y[2])**2)

def cost_control(u, yaw_t, weight_vel = 16, weight_yaw = 0, vel_target = 2, dt = 0.05):

    # vel_norm = np.linalg.norm(np.subtract(u[:-1], [0.0001, 0.0001, 0.0001]))
    # cost = weight_vel*(vel_norm - vel_target)**2
    cost =  weight_vel*((u[0] - 2)**2 + u[1]**2 + u[2]**2)
    # difference = abs(np.arctan2(u[1], u[0])%(2math.pi) - yaw_t%(2*math.pi))
    # difference = min(difference, 2*math.pi-difference)
    # cost += weight_yaw*(difference/dt - u[3])**2
    # print min(difference, 2*math.pi-difference)
    return cost
    
def gradient_control(func, x_0, epsilon = 1e-5):
    gradient = np.zeros([len(x_0)])
    for i in xrange(len(x_0)):
        x_0[i] += epsilon
        f1 = func(x_0)
        x_0[i] -= 2*epsilon
        f2 = func(x_0)
        x_0[i] += epsilon
        gradient[i] = (f1-f2)/(2*epsilon)
    return gradient

def hessian_control(func, x_0, epsilon = 1e-5):
    hes = np.zeros([len(x_0),len(x_0)])
    for i in xrange(len(x_0)):
        # import IPython; IPython.embed()
        x_0[i] += epsilon
        f1 = gradient_control(func, x_0)
        x_0[i] -= 2*epsilon
        f2 = gradient_control(func, x_0)
        x_0[i] += epsilon
        hes[i,:] = (f1-f2)/(2*epsilon)
    return hes


def yaw_to_quad(yaw):
    return single_yaw_to_quad(yaw)
        

def single_yaw_to_quad(yaw):
    return generate_orientation(yaw)
def quad_to_yaw(quad):
    if len(np.asarray(quad).shape) ==1:
        return single_quad_to_yaw(quad)
    else:
        length = np.asarray(quad).shape[0]
        desired_yaw =  np.zeros([length,1])
        for i in xrange(length):
            desired_yaw[i] = single_quad_to_yaw(quad[i,:])
        return desired_yaw

def single_quad_to_yaw(quad):
    m = tft.quaternion_matrix(quad)
    # import IPython;IPython.embed();
    theta_cos= math.acos(m[0, 0])
    theta_sin = math.asin(m[1, 0])
    if theta_sin > 0:
        return theta_cos
    else: return - theta_cos

def update_state(state):
    state[-1] = state[-1]%(math.pi*2)
    return state
def vyaw_from_yaw(yaw, dt = 0.05):
    if len(yaw) == 1:
        return 0.0
    else:
        desired_vel = np.zeros(len(yaw))
        desired_vel[:-1] = np.subtract(yaw[1:],yaw[:-1])/dt
        desired_vel[-1] = desired_vel[-2]
    return desired_vel
def circular_state(t, height, T):
    # import IPython; IPython.embed()
    desired_path, desired_yaw = circular_path_state(t, height, T)
    return np.concatenate([desired_path, desired_yaw], axis =1)

def yaw_optimize(sample, dt = 0.05):

    states = sample.get_X()
    controls = sample.get_U()
    for i in xrange(sample._T - 1):
        yaw_t = states[i, - 1]
        # yaw_t = np.arctan2(controls[i, 1],controls[i, 0])
        if yaw_t >math.pi or yaw_t < - math.pi:
            yaw_t = yaw_t%(2*math.pi)
        yaw_t1 = np.arctan2(controls[i + 1, 1],controls[i + 1, 0])
        # yaw_t1 = states[i + 1, - 1]
        if yaw_t1 > math.pi or yaw_t1 < - math.pi:
            yaw_t1 = yaw_t1%(2*math.pi)
        diff = yaw_t1 - yaw_t
        # if abs(diff) > 0.5:
        #     import IPython; IPython.embed()
        if  abs(diff) < math.pi:
            controls[i, -1] = diff/dt
        elif diff < -math.pi:
            controls[i, -1] = (diff + 2*math.pi)/dt
        else:
            controls[i, -1] = (diff - 2*math.pi)/dt
        sample.set_U(controls[i,:], t = i)
def u_t_optimize(x_t,u_t, u_t1, dt = 0.05):
    yaw_t = x_t[-1]
    if yaw_t >math.pi or yaw_t < - math.pi:
        yaw_t = yaw_t%(2*math.pi)
    yaw_t1 = np.arctan2(u_t1[1], u_t1[0])
        # yaw_t1 = states[i + 1, - 1]
    if yaw_t1 > math.pi or yaw_t1 < - math.pi:
        yaw_t1 = yaw_t1%(2*math.pi)
    diff = yaw_t1 - yaw_t
    if  abs(diff) < math.pi:
        u_t[-1] = diff/dt
    elif diff < -math.pi:
        u_t[-1] = (diff + 2*math.pi)/dt
    else:
        u_t[-1] = (diff - 2*math.pi)/dt
    return u_t
def test_control1(t, T):
    desired_control = np.zeros([T, 4])
    if t < 150:
        desired_control[:, 0] = 2
    else:
        desired_control[:, 1] = -2
    return desired_control

def init_desired_vel(x_0, x_t, x_t1, u_t, u_t1, env, dt = 0.2):
    u_optimized = u_t_optimize(x_t,u_t, u_t1)
    yaw_centered = u_optimized[-1]
    best_x = None
    best_dist = 10e5
    desired_delta_yaw = math.pi
    # yaw_range = np.concatenate([[0],np.linspace(-math.pi/3,math.pi/3, 7)])
    yaw_range = [0, -math.pi/9, math.pi/9, -math.pi/6, math.pi/6, -math.pi/3, math.pi/3,-4*math.pi/9,4*math.pi/9]
    for delta_yaw in yaw_range:
        yaw = yaw_centered + delta_yaw
        quad = yaw_to_quad(yaw)
        x_next = x_t1[:-1]
        x_next[0] += 2*math.cos(yaw)*dt
        x_next[1] += 2*math.sin(yaw)*dt
        env.rave_env.plot_point(x_next)
        pose = posquat_to_pose(x_next, quad)
        dist = best_dist
        if env.rave_env.closest_collision(pose) is None:
            desired_delta_yaw = delta_yaw
            best_x = x_next
            break
            # desired_vel_yaw = np.arctan2(best_x[1]-x_0[1],best_x[0] - x_0[0])
            # desired_vel = [2*math.cos(desired_vel_yaw),2*math.sin(desired_vel_yaw),u_t1[2]]
            # print desired_vel
            # return desired_vel
        _, dist, _, _ = env.rave_env.closest_collision(pose)
        # print dist
        if dist > 1:
            desired_delta_yaw = delta_yaw
            best_x = x_next
            break
            # desired_vel_yaw = np.arctan2(best_x[1]-x_0[1],best_x[0] - x_0[0])
            # desired_vel = [2*math.cos(desired_vel_yaw),2*math.sin(desired_vel_yaw),u_t1[2]]
            # print desired_vel
            # return desired_vel
        if best_x is None:
            desired_delta_yaw = delta_yaw
            best_x = x_next
            best_dist = dist
        if dist < best_dist:
            desired_delta_yaw = delta_yaw
            best_x = x_next
            best_dist = dist
    desired_vel_yaw = np.arctan2(best_x[1]-x_0[1],best_x[0] - x_0[0])
    desired_vel_yaw = round(200*desired_vel_yaw)/200
    desired_vel = [2*math.cos(desired_vel_yaw),2*math.sin(desired_vel_yaw),u_t1[2]]
    print 'desired_yaw', desired_vel_yaw
    # print desired_vel[0]
    return desired_vel


def quaternion_multiply(q1, q2, q3):
    # g1 = quaternion_matrix([q1[1],q1[2],q1[3],q1[0]])
    # g2 = quaternion_matrix([q2[1],q2[2],q2[3],q2[0]])
    # g3 = quaternion_matrix([q3[1],q3[2],q3[3],q3[0]])
    g1 = quaternion_matrix(q1)
    g2 = quaternion_matrix(q2)
    g3 = quaternion_matrix(q3)
    print np.dot(g3, g2).dot(g1)

def pos_in_drone_view(cur_pos, target_pos, yaw):
    c = np.array([cur_pos[0],cur_pos[1],cur_pos[2]]).T
    t = np.array([target_pos[0],target_pos[1],target_pos[2]]).T
    R = np.array([[math.cos(-yaw), -math.sin(-yaw), 0], [math.sin(-yaw), math.cos(-yaw), 0], [0, 0, 1]])
    t_new = np.dot(R, t)
    c_new = np.dot(R, c)
    return c_new, t_new

def quad_to_pitch(quad):
    m = tft.quaternion_matrix(quad)
    # print m
    # import IPython;IPython.embed();
    theta_sin= math.asin(m[0, 0])
    return -theta_sin
    
