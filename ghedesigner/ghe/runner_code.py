from ghedesigner.ghe.District_system_class import GHEHPSystem
from ghedesigner.ghe.HP_hybrid_loads_processor import ProcessLoads
import json
from OpenGL_2D_class_GLFW import gl2D, gl2DCircle, gl2DText,gl2DArrow, gl2DArc
import time

def main():
    f1 = open("input_files/Hybrid_Real_system_input.txt", 'r')
    data = f1.readlines()  # read the entire file as a list of strings
    f1.close()  # close the file  ... very important

    f2 = open("input_files/BALTIMORE_find_design_bi_rectangle_single_u_tube.json", 'r')
    json_data = json.load(f2)

    start_time = time.time()

    System = GHEHPSystem()

    hybrid_start = time.time()
    # Generate HP hybrid loads
    hybrid_system = ProcessLoads()
    hybrid_system.read_HP_load(data)
    if hybrid_system.method == "HYBRID":
        hybrid_system.run_hybrid_pipeline(json_data)

        # Pass hybrid results into simulation
        System.hybrid_processor = hybrid_system

    hybrid_end = time.time()

    print(f"Hybrid load generation time: {hybrid_end - hybrid_start:.4f} seconds")

    # Running main simulation
    System.read_GHEHPSystem_data(data)
    System.read_data_from_json_file(json_data)
    fluid, pipe, grout, soil, borehole, sim_params = System.read_data_from_json_file(json_data)
    System.solveSystem(fluid, pipe, grout, soil, borehole, sim_params)
    end_time = time.time()

    System.createOutput()
    System.output_file_energy_consumption()

    # Draw
    gl2d = gl2D(None, System.drawnetwork, width=2000, height=1500)
    gl2d.setViewSize(-10, 250, -10, 200, False)
    gl2d.glWait()  # wait for the user to close the window

    print(f"Execution time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()





