from ghedesigner.ghe.District_system_class import GHEHPSystem
import json
from OpenGL_2D_class_GLFW import gl2D, gl2DCircle, gl2DText,gl2DArrow, gl2DArc
import time

System = GHEHPSystem()

def main():
    f1 = open("input_files/Real_system_input.txt", 'r')
    data = f1.readlines()  # read the entire file as a list of strings
    f1.close()  # close the file  ... very important

    f2 = open("input_files/find_design_bi_rectangle_single_u_tube.json", 'r')
    json_data = json.load(f2)

    start_time = time.time()
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

    #print("Finished drawing 1")
    print(f"Execution time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()





