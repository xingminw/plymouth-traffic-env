import os
import sys
from shutil import copyfile, rmtree

# customize your own sumo env address!!
# for windows system, choose the place you install sumo
# TODO: add auto-recognition of the sumo folder path
# os.environ["SUMO_HOME"] = "C://Program Files (x86)//Eclipse//Sumo"
os.environ["SUMO_HOME"] = "/home/xingminw/Anaconda3/envs/traffic-env"

# gui mode, true to use traci, false to use libsumo, check your installment of libsumo
GUI_MODE = True

# random mode, true to use random demand, false to use the fixed random seed
RANDOM_MODE = True

IGNORE_WARNING = True
TELEPORT_MODE = False

output_folder = "output"
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

output_weights_folder = os.path.join(output_folder, "params")
if not os.path.exists(output_weights_folder):
    os.mkdir(output_weights_folder)

network_topology_folder = "sumo"
network_xml_file_template = "./sumo/plymouthv8.net.xml"
network_xml_file = "network.xml"
network_flow_file = "./sumo/FlowProperty.xml"
configuration_file_template = "./sumo/plymouthv8Routes.sumocfg"
turning_ratio_file = "./sumo/TurningRatio.turns.xml"
sumo_configuration_buffer_folder = "sumo/buffer"

route_xsd_file = "route.xsd"
network_xsd_file = "net.xsd"
buffer_route_head = "route_"
buffer_configuration_file = os.path.join(sumo_configuration_buffer_folder, "configure_")

tuned_parameter_json_file = "./sumo/tuned.json"

# create new empty buffer folder
if os.path.exists(sumo_configuration_buffer_folder):
    rmtree(sumo_configuration_buffer_folder)
    os.mkdir(sumo_configuration_buffer_folder)
else:
    # create buffer file
    os.mkdir(sumo_configuration_buffer_folder)

# copy the two rsd file to buffer
copyfile(os.path.join(network_topology_folder, route_xsd_file),
         os.path.join(sumo_configuration_buffer_folder, route_xsd_file))
copyfile(os.path.join(network_topology_folder, network_xsd_file),
         os.path.join(sumo_configuration_buffer_folder, network_xsd_file))

# copy the network file to buffer
buffer_network_xml = os.path.join(sumo_configuration_buffer_folder, network_xml_file)
copyfile(network_xml_file_template, buffer_network_xml)

output_figure_folder = "output/figs"
output_data_folder = "output/data"
output_observation_mapping_file = "sumo/observation_guidance.json"

turning_ratio = {"62606176": {0: 0.224, 1: 0.708 / 2, 2: 0.708 / 2, 3: 0.068, 4: 0.522, 5: 0.252, 6: 0.226,
                              7: 0.045, 8: 0.905 / 2, 9: 0.905 / 2, 10: 0.05, 11: 0.133, 12: 0.150,
                              13: 0.717 / 2, 14: 0.717 / 2},
                 "62500824": {4: 0.1, 5: 0.711 / 2, 6: 0.711 / 2, 7: 0.189, 8: 0.377, 9: 0.343 / 2, 10: 0.343 / 2,
                              11: 0.280, 12: 0.176, 13: 0.812 / 2, 14: 0.812 / 2, 15: 0.012, 0: 0.03, 1: 0.526 / 2,
                              2: 0.526 / 2, 3: 0.444},
                 "62477148": {0: 0.104, 1: 0.880 / 2, 2: 0.880 / 2, 3: 0.016, 4: 0.190, 5: 0.362, 6: 0.448, 7: 0.033,
                              8: 0.774 / 2, 9: 0.774 / 2, 10: 0.193, 11: 0.526, 12: 0.051, 13: 0.423},
                 "62532012": {0: 0.031, 1: 0.969 / 2, 2: 0.969, 3: 0.941 / 2, 4: 0.941 / 2, 5: 0.059,
                              6: 0.622, 7: 0.378},
                 "767530322": {0: 0.024, 1: 0.928 / 2, 2: 0.928 / 2, 3: 0.048, 4: 0.350, 5: 0.055, 6: 0.595, 7: 0.04,
                               8: 0.954 / 2, 9: 0.954 / 2, 10: 0.006, 11: 0.286, 12: 0.111, 13: 0.603},
                 "62500567": {0: 0.396, 1: 0.604 / 2, 2: 0.604 / 2, 3: 0.819 / 2,
                              4: 0.819 / 2, 5: 0.181, 6: 0.27, 7: 0.73}}

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'...")


sumoBinary = os.path.join(os.environ['SUMO_HOME'], 'bin/sumo')
