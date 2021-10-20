main.py: train the whole system
joint_angle_predictor.py: train the image encoder to predict joint angles
main_joints_predictor.py: train the whole system with an auxilliary loss os joint angles
main_small_image.py: cortex mode with 16x16 image
main_supervised_attn.py: cortex mode with supervised attention on target position prediction
main_cortex2.py: cortex mode with more trials. weight decay, more supervision on attention map, more layers, etc.
main_full_cortex.py: cortex mode with seperate objective and subjective parts
backbone_full_cortex.py goes accidently to empty file (system failure)
now cortex mode with seperate objective and subjective parts go to full_cortex2
