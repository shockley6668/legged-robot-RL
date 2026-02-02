import os

ROOT_DIR = os.path.dirname(__file__)
ENVS_DIR = os.path.join(ROOT_DIR,'Env')
#Taitan Tinker Tinymal
ROBOT_SEL = 'Tinker'
#Trot Stand
GAIT_SEL = 'Trot'
PLAY_DIR = os.path.join(ROOT_DIR, 'modelt_test.pt')
#Sim2Sim Cmd
SPD_X = 0.0
SPD_Y = 0.0
SPD_YAW = 0


#train param
MAX_ITER = 30000
SAVE_DIV = 500


#./compile XX.urdf XX.xml
#rosrun robot_state_publisher robot_state_publisher my_robot.urdf