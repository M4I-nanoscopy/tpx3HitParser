import sys
import lib
from events import chip_edge_correct

io = lib.io()

events = io.read_events(sys.argv[1])
new_events = chip_edge_correct(events)
io.open_write(sys.argv[2])
io.write_event_chunk(new_events, False)
io.store_events('cnn', 'cnn-tot-chip-edge', 2, events.attrs['min_toa'], events.attrs['max_toa'])
io.close_write()