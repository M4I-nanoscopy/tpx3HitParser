import numpy

# Hit matrix data type
dt_hit = numpy.dtype([
    ('chipId', numpy.uint8),
    ('x', numpy.uint8),
    ('y', numpy.uint8),
    ('ToT', numpy.uint8),
    ('cToA', numpy.uint16),
    ('TSPIDR', numpy.uint16)
])

# Cluster info key indeces
dt_ci = numpy.dtype([
    ('chipId', numpy.uint8),
    ('x', numpy.uint8),
    ('y', numpy.uint8),
    ('cToA', numpy.uint16),
    ('TSPIDR', numpy.uint16)
])

# Event matrix key indeces
dt_event = numpy.dtype([
    ('chipId', numpy.uint8),
    ('x', numpy.float64),
    ('y', numpy.float64),
    ('cToA', numpy.uint16),
    ('TSPIDR', numpy.uint16),
])

# Control events
CONTROL_END_OF_COMMAND = 0x71bf # 29119
CONTROL_END_OF_READOUT = 0x71b0 # 29104
CONTROL_END_OF_SEQUANTIAL_COMMAND = 0x71ef # 29167
CONTROL_OTHER_CHIP_COMMAND = 0x7200 # 29184
