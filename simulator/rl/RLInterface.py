import os
import csv
from simulator.core.DtnCore import Simulable


class RLInterface(Simulable):

    def __init__(self, env, node_ids, delta_t=10, control_dt = 10, outfile="rl_states", view_onscreen=True):
        super().__init__(env)

        self.env = env
        self.delta_t = delta_t
        self.view_onscreen = view_onscreen
        self.nodes = {nid: env.nodes[nid] for nid in node_ids}
        self.prev_arrivals = {nid: 0 for nid in self.nodes}
        self.prev_arrival_bytes = {nid: 0 for nid in self.nodes}
        self.prev_departures = {}
        self.control_dt = control_dt
        self.last_control_time = 0


        outdir = env.config['globals'].outdir
        self.outdir = outdir
        self.outfile_prefix = outfile
        os.makedirs(outdir, exist_ok=True)

        self.files = {}
        for nid in self.nodes:
            file_path = os.path.join(outdir, f"{outfile}_{nid}.csv")
            self.files[nid] = file_path

            with open(file_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["time", "queue_size_earth", "queue_size_relay", "queue_size_mars","radio_in_queue", "node_in_queue", "node_limbo_queue","node_total_induct", "node_total_outduct", "earth_conn", "relay_conn", "mars_conn","departure", "departure_rate","departure_in_bytes","departure_rate_in_Bps","arrivals","arrival_rate","arrival_in_bytes","arrival_rate_in_Bps"])

        print(f"[RL] Logging directory: {outdir}")
        print(f"[RL] Nodes attached: {list(self.nodes.keys())}")
        print(f"[RL] Sampling every {delta_t} seconds")

        self.samples = 0
        self.env.process(self.run())

    def get_bundle_size(self, node):
        bundle_size = 0
        if hasattr(node, 'generators'):
            for bundle in node.generators.values():
                if hasattr(bundle, 'bundle_sz'):
                    bundle_size = bundle.bundle_sz                   
                else:
                    bundle_size = 0
        return bundle_size   

    def get_queue_sizes(self, node):

        total_earth = 0
        total_relay = 0
        total_mars = 0
        node_in_queue = 0
        node_limbo_queue = 0
        total_outduct = 0
        total_induct = 0
        
        if hasattr(node, 'queues'):
        
            for neighbor, q in node.queues.items():
            
                if q is None or not hasattr(q, 'stored'):
                    continue

                size = len(q.stored)

                if neighbor == 'EARTH':
                    total_earth += size
                    
                elif neighbor == 'RELAY':
                    total_relay += size
                    
                elif neighbor == 'MARS':
                    total_mars += size
        else:
            print("[RL] Error: node has no attribute to queues to queues")
            
        if hasattr(node, 'in_queue') and hasattr(node.in_queue, 'items'):
            node_in_queue += len(node.in_queue.items)
            
        else:
            print("[RL] Error: node has no Attribute to in_queue or node.in_queue has no attribute to items")
            
        if hasattr(node, 'limbo_queue') and hasattr(node.limbo_queue, 'items'):
            node_limbo_queue += len(node.limbo_queue.items) 
        
        else:
            print("[RL] Error: node has no Attribute to limbo_queue")
            
        if hasattr(node, 'ducts'):
            for neighbor, bands in node.ducts.items():
                for band, duct_types in bands.items():
                    
                    if 'outduct' in duct_types:
                        d = duct_types['outduct']
                        if hasattr(d, 'stored'):
                            total_outduct += len(d.stored)
                        else:
                            print("[RL] Error: outducts have no attribute to stored")

                    if 'induct' in duct_types:
                        d = duct_types['induct']
                        if hasattr(d, 'stored'):
                            total_induct += len(d.stored)
                        else:
                            print("[RL] Error: inducts have no attribute to stored")
        else:
            print("[RL] Error: node has no attribute to ducts")

            
        return total_earth, total_relay, total_mars, node_in_queue, node_limbo_queue, total_induct, total_outduct

    def get_contact_states(self, nid):
        contact_states = {}
        
        for (origin, destination), conn in self.env.connections.items():
            
            if origin == nid:
                neighbor = destination
            elif destination == nid:
                neighbor = origin
            else:
                continue
                
            active = int(conn.active)
            contact_states[neighbor] = active
                 
        return contact_states
    
    def complete_contact_state(self, nid, connection_states):
        full_contact_state = {}

        for node_name in self.env.nodes.keys():

            if node_name == nid:
                full_contact_state[node_name] = "Self"

            else:
                full_contact_state[node_name] = connection_states.get(node_name, 0)

        return full_contact_state
        
    def get_departure_rate(self, node_id):

        departures = 0
        departures_in_bytes = 0

        for (origin, destination), conn in self.env.connections.items():

            if origin != node_id:
                continue

            sent_dict = conn.sent
            key = (origin, destination)

            prev = self.prev_departures.get(key, 0)
            current = len(sent_dict)
           
            if current > prev:
                new_records = list(sent_dict.values())[prev:current]

                for rec in new_records:
                    departures += 1
                    departures_in_bytes += rec.get('dv', 0)

            self.prev_departures[key] = current

        dep_rate = departures / self.delta_t
        dep_rate_in_bytes = departures_in_bytes / self.delta_t

        return departures, dep_rate, departures_in_bytes, dep_rate_in_bytes

    
    def get_arrival_rate(self, nid):
        node = self.nodes[nid] 
        
        current_count = node.arrivals_count
        arrivals = current_count - self.prev_arrivals[nid]
        self.prev_arrivals[nid] = current_count
        
        current_bytes = node.arrivals_bytes_count
        arrivals_in_bytes = current_bytes - self.prev_arrival_bytes[nid]
        self.prev_arrival_bytes[nid] = current_bytes

        arr_rate = arrivals / self.delta_t
        arr_rate_in_Bps = arrivals_in_bytes / self.delta_t
        
        return arrivals, arr_rate, arrivals_in_bytes, arr_rate_in_Bps
    
#============================================= RL AGENT ==========================================

    def dummy_agent(self, nid, state):
        radio_in_queue = state["radio_in_queue"]

        if radio_in_queue > 1174:
            return 1.5e6   # 5 Mbps
        else:
            return 2.5e6    # 2 Mbps
        
#============================================= RL AGENT === APPLY ACTION ========================================
    
    def apply_rate(self, node, new_rate):
        if hasattr(node, 'radios'):
            for rid, radio in node.radios.items():
                radio.datarate = new_rate
        else:
            print("[RL] Error: No Attribute to radios")
    def run(self):

        while True:
        
            t = self.env.now
            
            if self.env.now == 0:
                for nid, node in self.env.nodes.items():
                    print(nid, type(node.router))


            for nid, node in self.nodes.items():
                total_earth, total_relay, total_mars, node_in_queue, node_limbo_queue, total_induct, total_outduct = self.get_queue_sizes(node)
                
                contacts = self.get_contact_states(nid)
                full_contact_state = self.complete_contact_state(nid, contacts)
                departures, dep_rate, departures_in_bytes, dep_rate_in_bytes = self.get_departure_rate(nid)
                arrivals, arr_rate, arrivals_in_bytes, arr_rate_in_Bps = self.get_arrival_rate(nid)
                 
                rate = 0 
                radio_in_queue = 0
                
                for rid, radio in node.radios.items():
                    radio_in_queue = len(radio.in_queue.items)  
                     
                if nid == "RELAY":
                    state = {"radio_in_queue": radio_in_queue} 
                    
                    if self.env.now - self.last_control_time >= self.control_dt:
                        action_rate = self.dummy_agent(nid, state) 
                        self.apply_rate(node, action_rate)
                        self.last_control_time = self.env.now     
                                        
                if self.view_onscreen:
                    print(f"[{t}] Node RLI Attached: {nid} -> E:{total_earth} R:{total_relay} M:{total_mars}")
                    print(f"[{t}] Node In Queue: {node_in_queue}, Node Limbo Queue: {node_limbo_queue}")
                    print(f"[{t}] Node Total Induct: {total_induct}, Node Total Outduct: {total_outduct}")
                    print(f"[{t}] Departures: {departures_in_bytes} Bytes and Departure Rate: {dep_rate_in_bytes} Bps")
                    print(f"[{t}] Arrivals: {arrivals_in_bytes} Bytes abd Arrival Rate: {arr_rate_in_Bps}")
                    print()
                
                with open(self.files[nid], "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([t, total_earth, total_relay, total_mars, radio_in_queue, node_in_queue, node_limbo_queue, total_induct, total_outduct, full_contact_state['EARTH'], full_contact_state['RELAY'], full_contact_state['MARS'], departures, dep_rate, departures_in_bytes, dep_rate_in_bytes, arrivals, arr_rate, arrivals_in_bytes, arr_rate_in_Bps])

            self.samples += 1

            if self.view_onscreen:
                print()
                print()

            yield self.env.timeout(self.delta_t)


    def finalize(self):
        print(f"[RL] Finished. {self.samples} samples saved.")
        for nid, path in self.files.items():
            print(f"[RL] {nid}: {path}")

