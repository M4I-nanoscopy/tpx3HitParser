import numpy

VERSION = '2.0.0'

# Hit matrix data type
dt_hit = numpy.dtype([
    ('chipId', numpy.uint8),
    ('x', numpy.uint16),
    ('y', numpy.uint16),
    ('ToT', numpy.uint16),
    ('ToA', numpy.int64)
])
HITS_CHUNK_SIZE = 10_000_000

# Cluster info key indices
dt_ci_base = [
    ('chipId', numpy.uint8),
    ('x', numpy.uint16),
    ('y', numpy.uint16),
    ('ToA', numpy.int64),
]
dt_ci_extended = [
    ('sumToT', numpy.uint16),
    ('nHits', numpy.uint8)
]

dt_clusters = 'uint16'

CLUSTER_CHUNK_SIZE = 100_000

# Event matrix key indices
dt_event_base = [
    ('chipId', numpy.uint8),
    ('x', numpy.float64),
    ('y', numpy.float64),
    ('ToA', numpy.int64),
]
dt_event_extended = [
    ('sumToT', numpy.uint16),
    ('nHits', numpy.uint8)
]

EVENTS_CHUNK_SIZE = 100_000

# Control events
CONTROL_END_OF_COMMAND = 0x71bf # 29119
CONTROL_END_OF_READOUT = 0x71b0 # 29104
CONTROL_END_OF_SEQUANTIAL_COMMAND = 0x71ef # 29167
CONTROL_OTHER_CHIP_COMMAND = 0x7200 # 29184

# ToT correction matrix shape
TOT_CORRECTION_SHAPE = (1024, 256, 256, 4)
