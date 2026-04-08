import os

ROOT_DIR = os.path.dirname(__file__)
ENVS_DIR = os.path.join(ROOT_DIR,'Env')
#Taitan Tinker Tinymal
ROBOT_SEL = 'Tinker'
#Trot Stand
GAIT_SEL = 'Trot'
# PLAY_DIR = os.path.join(ROOT_DIR, 'modelt_test.pt')
PLAY_DIR = "/home/fsr/legged-robot-RL/logs/rough_go2_constraint/Mar18_14-00-48_test_barlowtwins_phase2/model_1000.pt"
#Sim2Sim Cmd
SPD_X = 0.0
SPD_Y = 0.0
SPD_YAW = 0


#train param
MAX_ITER = 13000
SAVE_DIV = 1000


#./compile XX.urdf XX.xml
#rosrun robot_state_publisher robot_state_publisher my_robot.urdf