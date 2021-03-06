import numpy

VERSION = '1.0.0'

# Hit matrix data type
dt_hit = numpy.dtype([
    ('chipId', numpy.uint8),
    ('x', numpy.uint16),
    ('y', numpy.uint16),
    ('ToT', numpy.uint16),
    ('cToA', numpy.uint16),
    ('TSPIDR', numpy.uint16),
    ('fToA', numpy.uint8),
    ('spId', numpy.uint16),
    ('pix', numpy.uint8)
])
HITS_CHUNK_SIZE = 10000000

# Cluster info key indices
dt_ci = numpy.dtype([
    ('chipId', numpy.uint8),
    ('x', numpy.uint16),
    ('y', numpy.uint16),
    ('cToA', numpy.uint16),
    ('TSPIDR', numpy.uint16),
    ('sumToT', numpy.uint16)
])
dt_clusters = 'uint16'

CLUSTER_CHUNK_SIZE = 100000

# Event matrix key indices
dt_event = numpy.dtype([
    ('chipId', numpy.uint8),
    ('x', numpy.float64),
    ('y', numpy.float64),
    ('cToA', numpy.uint16),
    ('TSPIDR', numpy.uint16),
    ('sumToT', numpy.uint16)
])

# Control events
CONTROL_END_OF_COMMAND = 0x71bf # 29119
CONTROL_END_OF_READOUT = 0x71b0 # 29104
CONTROL_END_OF_SEQUANTIAL_COMMAND = 0x71ef # 29167
CONTROL_OTHER_CHIP_COMMAND = 0x7200 # 29184
