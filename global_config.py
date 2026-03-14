import os

ROOT_DIR = os.path.dirname(__file__)
ENVS_DIR = os.path.join(ROOT_DIR,'Env')
#Taitan Tinker Tinymal
ROBOT_SEL = 'Tinker'
#Trot Stand
GAIT_SEL = 'Trot'
# PLAY_DIR = os.path.join(ROOT_DIR, 'modelt_test.pt')
PLAY_DIR = "/home/fsr/Downloads/OmniBotSeries-Tinker/OmniBotCtrl/OmniBotCtrl/logs/rough_go2_constraint/Feb02_20-27-21_test_barlowtwins/model_26000.pt"
#Sim2Sim Cmd
SPD_X = 0.0
SPD_Y = 0.0
SPD_YAW = 0


#train param
MAX_ITER = 16000
SAVE_DIV = 500


#./compile XX.urdf XX.xml
#rosrun robot_state_publisher robot_state_publisher my_robot.urdf