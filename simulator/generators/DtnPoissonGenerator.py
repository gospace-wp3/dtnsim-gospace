import numpy as np
import random
from simulator.core.DtnBundle import Bundle
from simulator.generators.DtnAbstractGenerator import DtnAbstractGenerator


class DtnPoissonGenerator(DtnAbstractGenerator):

    def initialize(self):
    
        self.lambda_rate = self.props.lambda_rate

        self.tstart = self.props.tstart
        self.tend   = self.props.tend

        self.data_type  = self.props.data_type
        self.bundle_sz  = self.props.bundle_size
        self.bundle_lat = self.props.bundle_TTL
        self.critical   = self.props.critical

        self.orig = self.parent.nid
        self.dest = self.props.destination

        self.generated_bundles = 0

        self.env.process(self.run())

    def run(self):

        yield self.env.timeout(self.tstart)

        nodes = list(self.env.nodes.keys())
        nodes.remove(self.parent.nid)

        while self.env.now < self.tend:

            # --- Exponential inter-arrival ---
            inter_arrival = random.expovariate(self.lambda_rate)
            yield self.env.timeout(inter_arrival)

            # Stop if simulation time exceeded
            #if self.env.now >= self.tend:
            #    break

            if self.dest:
                dest = self.dest
            else:
                dest = random.choice(nodes)

            new_bundle = self.new_bundle(dest)

            self.generated_bundles += 1
            self.monitor_new_bundle(new_bundle)
            self.disp('{} created at node {}', new_bundle, self.parent.nid)

            if not self.is_alive:
                yield self.env.exit()

            self.parent.forward(new_bundle)

    def predicted_data_vol(self):

        duration = max(0, self.tend - self.tstart)
        expected_bundles = self.lambda_rate * duration
        return expected_bundles * self.bundle_sz

    def new_bundle(self, dest):

        return Bundle(
            self.env,
            self.orig,
            dest,
            self.data_type,
            self.bundle_sz,
            self.bundle_lat,
            self.critical,
            fid=self.fid,
            TTL=self.bundle_lat
        )

