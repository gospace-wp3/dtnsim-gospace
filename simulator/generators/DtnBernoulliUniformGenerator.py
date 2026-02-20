import numpy as np
import random
from simulator.core.DtnBundle import Bundle
from simulator.generators.DtnAbstractGenerator import DtnAbstractGenerator


class DtnBernoulliUniformGenerator(DtnAbstractGenerator):

    def initialize(self):
    
        self.p = self.props.p 
        self.min_bundles = self.props.min_bundles             
        self.max_bundles = self.props.max_bundles              

        self.tstart = self.props.tstart
        self.tend   = self.props.tend

        self.data_type  = self.props.data_type
        self.bundle_sz  = self.props.bundle_size
        self.bundle_lat = self.props.bundle_TTL
        self.critical   = self.props.critical

        self.orig = self.parent.nid
        self.dest = self.props.destination

        self.env.process(self.run())

    def run(self):

        yield self.env.timeout(self.tstart)

        nodes = list(self.env.nodes.keys())
        nodes.remove(self.parent.nid)

        while self.env.now < self.tend:

            if self.dest:
                dest = self.dest
            else:
                dest = random.choice(nodes)

            # --- Bernoulli decision ---
            if random.random() < self.p:

                k = random.randint(self.min_bundles, self.max_bundles)

                for _ in range(k):
                    new_bundle = self.new_bundle(dest)

                    self.monitor_new_bundle(new_bundle)
                    self.disp('{} created at {}', new_bundle, self.parent.nid)

                    if not self.is_alive:
                        yield self.env.exit()

                    self.parent.forward(new_bundle)

            # --- Time Slot: 1 second ---
            yield self.env.timeout(1)
            
    def predicted_data_vol(self):
    
        expected_per_sec = self.p * (self.a + self.b) / 2
        duration = self.tend - self.tstart
        return expected_per_sec * duration * self.bundle_sz

    def new_bundle(self, dest):
        return Bundle(self.env, self.orig, dest, self.data_type,
                      self.bundle_sz, self.bundle_lat, self.critical,
                      fid=self.fid, TTL=self.bundle_lat)

