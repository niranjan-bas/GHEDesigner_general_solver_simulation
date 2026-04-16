import pandas as pd
import numpy as np
from ghedesigner.ghe.ground_loads import HybridLoad

from copy import deepcopy
from ghedesigner.ghe.coaxial_borehole import get_bhe_object
from ghedesigner.media import Grout, Soil, GHEFluid
from ghedesigner.media import Pipe as MediaPipe
from pygfunction.boreholes import Borehole
from ghedesigner.ghe.simulation import SimulationParameters
from ghedesigner.enums import BHPipeType

import json

class Zone:
    def __init__(self):
        self.q_htg = None
        self.q_clg = None
        self.q_ext = None
        self.q_htg_hybrid = None
        self.q_clg_hybrid = None
        self.q_ext_hybrid = None
        self.q_rej_hybrid = None
        self.q_ext_common = None
        self.q_rej_common = None
        self.q_ext_hybrid_time_array = None
        self.q_rej_hybrid_time_array = None
        self.q_htg_hybrid = None
        self.q_clg_hybrid = None
        self.hybrid_time_array = None   # this is heat pump common time array and is same for all zones (heat pumps)

        self.COP_clg = None
        self.COP_htg = None
        self.loads_file = None

    def initialize_load_arrays(self, n_years):
        """
        Read load columns once and store all arrays for later timestep access.
        """
        if "HPHtgLd_W" in self.loads_file.columns:
            h_1yr = self.loads_file["HPHtgLd_W"].to_numpy(dtype=float)
        else:
            h_1yr = np.zeros(len(self.loads_file), dtype=float)

        if "HPClgLd_W" in self.loads_file.columns:
            c_1yr = self.loads_file["HPClgLd_W"].to_numpy(dtype=float)
        else:
            c_1yr = np.zeros(len(self.loads_file), dtype=float)

        h_full = np.tile(h_1yr, n_years)
        c_full = np.tile(c_1yr, n_years)

        # Prepend zero at timestep 0
        self.q_htg = np.insert(h_full, 0, 0.0)
        self.q_clg = np.insert(c_full, 0, 0.0)

        return self.q_clg, self.q_htg

    def convert_HP_loads_to_ground_loads(self):
        self.q_rej = self.q_clg * (1 + 1/self.COP_clg)
        self.q_ext = self.q_htg * (1-1/self.COP_htg)

        return self.q_rej, self.q_ext

    def generate_hybrid_loads(self, bhe, radial_numerical, sim_params, years):
        ext_obj = HybridLoad(
            raw_loads=self.q_ext[1:],
            bhe=bhe,
            radial_numerical=radial_numerical,
            sim_params=sim_params,
            years=years,
        )

        rej_obj = HybridLoad(
            raw_loads=-self.q_rej[1:],
            bhe=bhe,
            radial_numerical=radial_numerical,
            sim_params=sim_params,
            years=years,
        )

        self.q_ext_hybrid = ext_obj.load[2:] * 1000
        self.q_ext_hybrid_time_array = ext_obj.hour[2:]
        self.q_rej_hybrid = rej_obj.load[2:] * 1000
        self.q_rej_hybrid_time_array = rej_obj.hour[2:]

    def map_loads_to_common_time(self, common_time):

        q_ext_time = self.q_ext_hybrid_time_array
        q_rej_time = self.q_rej_hybrid_time_array

        idx_ext = np.searchsorted(q_ext_time, common_time, side="left")
        idx_ext = np.clip(idx_ext, 0, len(self.q_ext_hybrid) - 1)
        self.q_ext_common = self.q_ext_hybrid[idx_ext]

        idx_rej = np.searchsorted(q_rej_time, common_time, side="left")
        idx_rej = np.clip(idx_rej, 0, len(self.q_rej_hybrid) - 1)
        self.q_rej_common = self.q_rej_hybrid[idx_rej]

        return self.q_rej_common, self.q_ext_common

    def convert_ground_hybrid_loads_to_HP_loads(self, common_time):
        self.q_htg_hybrid = self.q_ext_common/(1-1/self.COP_htg)*(-1)
        self.q_clg_hybrid = self.q_rej_common/(1+1/self.COP_clg)
        self.hybrid_time_array = common_time


class ProcessLoads:
    def __init__(self):
        self.n_years = None
        self.zones = []

        self.fluid = None
        self.pipe = None
        self.grout = None
        self.soil = None
        self.borehole = None
        self.sim_params = None

        self.bhe = None
        self.bhe_eq = None
        self.load_years = None

        self.mass_flow_rate = None
        self.flow_type = None

        self.common_time = None

    def read_data_from_json_file(self, json_data):
        self.data = json_data

        # Extract input values
        fluid_data = json_data["fluid"]
        soil_data = json_data["ground-heat-exchanger"]["ghe1"]["soil"]
        grout_data = json_data["ground-heat-exchanger"]["ghe1"]["grout"]
        pipe_data = json_data["ground-heat-exchanger"]["ghe1"]["pipe"]
        borehole_data = json_data["ground-heat-exchanger"]["ghe1"]["borehole"]
        geometric_data = json_data["ground-heat-exchanger"]["ghe1"]["geometric_constraints"]
        design_data = json_data["ground-heat-exchanger"]["ghe1"]["design"]
        num_months = json_data["simulation-control"]["simulation-months"]

        # Construct objects
        self.fluid = (
            GHEFluid(
                fluid_data["fluid_name"],
                fluid_data["concentration_percent"],
                fluid_data["temperature"]
            ))
        # Pipe object (Single U-tube)
        r_in = pipe_data["inner_diameter"] / 2.0
        r_out = pipe_data["outer_diameter"] / 2.0
        s = pipe_data["shank_spacing"]

        pipe_positions = MediaPipe.place_pipes(s, r_out, 1)

        self.pipe = MediaPipe(
            pipe_positions,
            r_in,
            r_out,
            s,
            pipe_data["roughness"],
            pipe_data["conductivity"],
            pipe_data["rho_cp"]
        )

        self.soil = Soil(soil_data["conductivity"], soil_data["rho_cp"], soil_data["undisturbed_temp"])
        self.grout = Grout(grout_data["conductivity"], grout_data["rho_cp"])
        self.borehole = Borehole(100, borehole_data["buried_depth"], borehole_data["diameter"] / 2.0, 0.0, 0.0)  # I assign height as 100 for all

        # mass flow rate
        self.mass_flow_rate = design_data["flow_rate"]
        self.flow_type = design_data["flow_type"]

        # Simulation parameters
        self.sim_params = SimulationParameters(num_months=num_months)
        self.sim_params.set_design_heights(geometric_data["max_height"], geometric_data["min_height"])
        self.sim_params.set_design_temps(design_data["max_eft"], design_data["min_eft"])

        return self.fluid, self.pipe, self.grout, self.soil, self.borehole, self.sim_params

    def read_HP_load(self, data):

        for line in data:  # loop over all the lines
            cells = [c.strip() for c in line.strip().split(',')]
            keyword = cells[0].lower()

            if keyword == "simulation_info":
                self.method = str(cells[1])
                self.n_years = int(cells[2])

            if keyword == 'zone':
                df = pd.read_csv(cells[7])
                self.time_array = df['Hours'].values.astype(float)         # this is same for all zones
                self.time_array_size = len(self.time_array)

                thiszone = Zone()
                thiszone.time_array = self.time_array
                thiszone.time_array_size = len(thiszone.time_array)
                thiszone.loads_file = df
                thiszone.COP_htg = float(cells[8])
                thiszone.COP_clg = float(cells[9])
                self.zones.append(thiszone)

    def prepare_bhe_for_hybrid(self):
        # example: this assumes these objects are already assigned
        borehole = deepcopy(self.borehole)

        bhe_type = BHPipeType.SINGLEUTUBE
        mass_flow_borehole = self.mass_flow_rate

        self.bhe = get_bhe_object(
            bhe_type,
            mass_flow_borehole,
            self.fluid,
            borehole,
            self.pipe,
            self.grout,
            self.soil,
        )

        self.bhe_eq = self.bhe.to_single()
        self.bhe_eq.calc_sts_g_functions()

        self.load_years = [2019]

    def generate_hybrid_ground_loads(self):
        for zone in self.zones:
            zone.q_clg, zone.q_htg = zone.initialize_load_arrays(self.n_years)
            zone.q_rej, zone.q_ext = zone.convert_HP_loads_to_ground_loads()
            zone.generate_hybrid_loads(bhe=self.bhe_eq, radial_numerical=self.bhe_eq, sim_params=self.sim_params, years=self.load_years)

    def generate_common_timegrid(self):
        all_times = np.concatenate([zone.q_ext_hybrid_time_array for zone in self.zones] +
                                       [zone.q_rej_hybrid_time_array for zone in self.zones])
        self.common_time = np.unique(all_times)
        self.common_time.sort()
        return self.common_time

    def map_all_zones(self):
        for zone in self.zones:
            zone.map_loads_to_common_time(self.common_time)

    def create_HP_hybrid_loads(self):
        for zone in self.zones:
            zone.convert_ground_hybrid_loads_to_HP_loads(self.common_time)

    def create_output_dataframe(self):
        df = pd.DataFrame()

        for i, zone in enumerate(self.zones, start=1):
            df[f"Zone{i}_Time"] = zone.hybrid_time_array
            df[f"Zone{i}_q_htg"] = zone.q_htg_hybrid
            df[f"Zone{i}_q_clg"] = zone.q_clg_hybrid

        self.output_df = df
        return df

    def write_hybrid_output_csv(self, output_file="results/hybrid_loads_data.csv"):
        if not hasattr(self, "output_df") or self.output_df is None:
            self.create_output_dataframe()

        df = self.output_df.copy()
        step_df = pd.DataFrame()

        for col in df.columns:
            values = df[col].to_numpy()

            repeated_values = []

            if "Time" in col:
                # time: [t0, t0, t1, t1, ...]
                for i in range(len(values)):
                    repeated_values.append(values[i])
                    repeated_values.append(values[i])

            else:
                # loads: [q0, q1, q1, q2, q2, q3, ...]
                for i in range(len(values) - 1):
                    repeated_values.append(values[i])
                    repeated_values.append(values[i + 1])

                # handle last value → repeat it
                repeated_values.append(values[-1])
                repeated_values.append(values[-1])

            step_df[col] = repeated_values

        step_df.to_csv(output_file, index=False)

    def run_hybrid_pipeline(self, json_data):
        self.read_data_from_json_file(json_data)
        self.prepare_bhe_for_hybrid()
        self.generate_hybrid_ground_loads()
        self.generate_common_timegrid()
        self.map_all_zones()
        self.create_HP_hybrid_loads()
        self.write_hybrid_output_csv()

