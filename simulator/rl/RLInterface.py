import os
import csv
from simulator.core.DtnCore import Simulable
from simulator.rl.RLAgent import RLAgent
from collections import deque

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
        self.prev_energy = {nid: 0 for nid in self.nodes}
        self.control_dt = control_dt
        self.last_control_time = 0
        self.drop_applied = False
        
        # Initializing for RL Agent
        self.agent = RLAgent()
        self.prev_state = None
        self.prev_action = None
        self.accum_reward = 0


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
                writer.writerow(["time", "queue_size_earth", "queue_size_relay", "queue_size_mars","radio_in_queue", "node_in_queue", "node_limbo_queue","node_total_induct", "node_total_outduct", "earth_conn", "relay_conn", "mars_conn","departure", "departure_rate","departure_in_bytes","departure_rate_in_Bps","arrivals","arrival_rate","arrival_in_bytes","arrival_rate_in_Bps","cummulative energy","energy_spent","power","rl_reward","dropped"])

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

               # size = len(q.stored)
                size = 0
                pq = q.queue
                
                for priority, dq in pq.items.items():
                    for rt_record in dq:
                        bundle = rt_record.bundle

                        if not bundle.dropped:
                            size += 1
                            
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

    def get_energy(self, nid):
        node = self.nodes[nid]
        current_energy = 0
        if hasattr(node, 'radios'):
            for rid, radio in node.radios.items():
                if hasattr(radio, 'energy'):
                    current_energy += radio.energy
                else:
                    print("[RL] Error: radios has no attribute to energy")
        else:
            print("[RL] Error: node has no attribute to radios")
            
        prev = self.prev_energy.get(nid,0)
        energy_change = current_energy - prev
        energy_rate = energy_change/self.delta_t
        self.prev_energy[nid] = current_energy
        return current_energy, energy_change, energy_rate       
    
    def get_bundle_info(self, node):

        bundle_info = []

        for neighbor, mgr in node.queues.items():

            if mgr is None or neighbor == 'opportunistic':
                continue

            for priority, q in mgr.queue.items.items():

                for rt_record in q:
                    bundle = rt_record.bundle
                    
                    if not bundle.dropped:
                        bundle_info.append({
                            "bid": bundle.bid,
                            "neighbor": neighbor,
                            "dest": bundle.dest,
                            "creation_time": bundle.creation_time,
                            "priority": priority
                        })

        return bundle_info
        
    def apply_drop_by_bid(self, node, bids_to_drop):

        bids_to_drop = set(bids_to_drop)
        dropped = 0

        for neighbor, mgr in node.queues.items():

            if mgr is None or neighbor == 'opportunistic':
                continue

            for priority, q in mgr.queue.items.items():

                for rt_record in list(q):
                    bundle = rt_record.bundle
                    
                    if bundle.bid in bids_to_drop and not bundle.dropped:
                    
                        node.drop(bundle, "RL Drop")
                        dropped += 1
                  
        print(f"[RL] Dropped {dropped} bundles")
        return dropped 
#============================================= RL AGENT ==========================================


        
#============================================= RL AGENT === APPLY ACTION ========================================
    
    def apply_rate(self, node, new_rate):
        if hasattr(node, 'radios'):
            for rid, radio in node.radios.items():
                radio.datarate = new_rate
        else:
            print("[RL] Error: No Attribute to radios")
    def run(self):

        while True:
            bundle_info = []
            t = self.env.now
            reward = 0
            if self.env.now == 0:
                for nid, node in self.env.nodes.items():
                    print(nid, type(node.router))

            for nid, node in self.nodes.items():
                total_earth, total_relay, total_mars, node_in_queue, node_limbo_queue, total_induct, total_outduct = self.get_queue_sizes(node)
                
                contacts = self.get_contact_states(nid)
                full_contact_state = self.complete_contact_state(nid, contacts)
                departures, dep_rate, departures_in_bytes, dep_rate_in_bytes = self.get_departure_rate(nid)
                arrivals, arr_rate, arrivals_in_bytes, arr_rate_in_Bps = self.get_arrival_rate(nid)
                cum_energy, energy_change, energy_rate = self.get_energy(nid)
                rate = 0 
                radio_in_queue = 0
                dropped = 0
                
                for rid, radio in node.radios.items():
                    radio_in_queue = len(radio.in_queue.items)  
                 
                if nid == "RELAY":
                
                    
                    #============DUMMY DROP =====================

               #    bundle_info.sort(key=lambda x: x["creation_time"])

               #    bids = [b["bid"] for b in bundle_info[:500]]

               #    self.apply_drop_by_bid(node, bids)

               #    self.drop_applied = True
                
                   #============DUMMY DROP =====================
                   
                    contact_state = full_contact_state['MARS']

                    state = {
                        "radio_queue": total_mars,
                        "departure_rate": dep_rate,
                        "energy_spent": energy_change,
                        "contact_state": contact_state
                    }

                    w_q = 0.02
                    w_e = 1.0
                    w_d = 0.1
                    w_dp = 1.0
                    
                    reward = (
                        - w_q * state["radio_queue"]
                        - w_e * state["energy_spent"]
                        + w_d * state["departure_rate"]
                        - w_dp * dropped
                    )

                    self.accum_reward += reward / 10


# Control interval
                    if self.env.now - self.last_control_time >= self.control_dt:

    # Learn from previous action
                        if self.prev_state is not None:
                            self.agent.learn(
                                self.prev_state,
                                self.prev_action,
                                self.accum_reward,
                                state
                            )

                        print(f"[RL] reward over interval = {self.accum_reward}")

    # --- New action ---
                        action_rate, drop_k = self.agent.act(state)

                        print(f"[RL] Action -> rate={action_rate/1e6} Mbps, drop={drop_k}")
                        print(f"[RL] Actual Dropped: {dropped}")

    # Apply rate
                        self.apply_rate(node, action_rate)

    # Apply dropping
                        if drop_k > 0:
                            bundle_info = self.get_bundle_info(node)
                            if bundle_info:
                                bundle_info.sort(key=lambda x: x["creation_time"])
                                bids_to_drop = [b["bid"] for b in bundle_info[:drop_k]]
                                dropped = self.apply_drop_by_bid(node, bids_to_drop)

    # Save for next step
                        self.prev_state = state
                        self.prev_action = (action_rate, drop_k)
                        self.accum_reward = 0
                        self.last_control_time = self.env.now
                                        
                if self.view_onscreen:
                    print(f"[{t}] Node RLI Attached: {nid} -> E:{total_earth} R:{total_relay} M:{total_mars}")
                    print(f"[{t}] Node In Queue: {node_in_queue}, Node Limbo Queue: {node_limbo_queue}")
                    print(f"[{t}] Node Total Induct: {total_induct}, Node Total Outduct: {total_outduct}")
                    print(f"[{t}] Departures: {departures_in_bytes} Bytes and Departure Rate: {dep_rate_in_bytes} Bps")
                    print(f"[{t}] Arrivals: {arrivals_in_bytes} Bytes abd Arrival Rate: {arr_rate_in_Bps}")
                    print(f"[{t}] Energy spent: {energy_change} J ; Power Measure: {energy_rate} W")
                    print(f"[{t}] RELAY bundles in neighbor queues: {len(bundle_info)}")
                    print()

                with open(self.files[nid], "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([t, total_earth, total_relay, total_mars, radio_in_queue, node_in_queue, node_limbo_queue, total_induct, total_outduct, full_contact_state['EARTH'], full_contact_state['RELAY'], full_contact_state['MARS'], departures, dep_rate, departures_in_bytes, dep_rate_in_bytes, arrivals, arr_rate, arrivals_in_bytes, arr_rate_in_Bps, cum_energy, energy_change, energy_rate,reward,dropped])

            self.samples += 1

            if self.view_onscreen:
                print()
                print()

            yield self.env.timeout(self.delta_t)


    def finalize(self):
        print(f"[RL] Finished. {self.samples} samples saved.")
        for nid, path in self.files.items():
            print(f"[RL] {nid}: {path}")

