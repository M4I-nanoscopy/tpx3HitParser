# Hit matrix indeces
CHIP = 0
X = 1
Y = 2
TOT = 3
TOA = 4
fTOA = 5
SPIDR_TIME = 6
cTOA = 7

# Cluster info key indeces
CI_CHIP = 0
CI_X = 1
CI_Y = 2
CI_SPIDR_TIME = 3
CI_cTOA = 4

# Event matrix key indeces
E_CHIP = 0
E_X = 1
E_Y = 2
E_TIME = 3

# Control events
CONTROL_END_OF_COMMAND = 0x71bf # 29119
CONTROL_END_OF_READOUT = 0x71b0 # 29104
CONTROL_END_OF_SEQUANTIAL_COMMAND = 0x71ef # 29167
CONTROL_OTHER_CHIP_COMMAND = 0x7200 # 29184


n_pixels = 6

cluster_chunk_size = 8000