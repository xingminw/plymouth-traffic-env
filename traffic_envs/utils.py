import os
import subprocess
import numpy as np
import xml.etree.ElementTree as ET

from traffic_envs import config


def generate_new_route_configuration(sumo_seed=None):
    """
    generate new route file...
    generate new configuration file and return the name...

    see link for the reference of jtrrouter: https://sumo.dlr.de/docs/jtrrouter.html
    :param sumo_seed:
    :return:
    """
    common_cmd = ["jtrrouter", "--flow-files=" + config.network_flow_file,
                  "--turn-ratio-files=" + config.turning_ratio_file,
                  "--net-file=" + config.network_xml_file_template]
    if config.RANDOM_MODE:
        if sumo_seed is not None:
            cmd_tail = ["--begin", "0", "--end", "3600", "--randomize-flows",
                        "--seed", str(sumo_seed), "--no-warnings"]
        else:
            cmd_tail = ["--begin", "0", "--end", "3600", "--randomize-flows", "--random", "--no-warnings"]
    else:
        cmd_tail = ["--begin", "0", "--end", "3600", "--randomize-flows"]

    route_xml_file = config.buffer_route_head + str(sumo_seed) + ".xml"
    buffer_route_xml_file = os.path.join(config.sumo_configuration_buffer_folder, route_xml_file)
    output_cmd = "--output-file=" + buffer_route_xml_file
    common_cmd.append(output_cmd)
    common_cmd += cmd_tail
    subprocess.run(common_cmd)

    while not os.path.exists(buffer_route_xml_file):
        pass

    route_tree = ET.parse(buffer_route_xml_file)
    for elem in route_tree.iter("routes"):
        elem.attrib["{http://www.w3.org/2001/XMLSchema-instance}noNamespaceSchemaLocation"] =\
            config.route_xsd_file
    route_tree.write(buffer_route_xml_file)

    configuration_tree = ET.parse(config.configuration_file_template)
    for elem in configuration_tree.iter("net-file"):
        elem.attrib["value"] = config.network_xml_file
    for elem in configuration_tree.iter("route-files"):
        elem.attrib["value"] = route_xml_file

    output_configuration_file = config.buffer_configuration_file + str(sumo_seed) + ".sumocfg"
    configuration_tree.write(output_configuration_file)

    load_vehicle_list = generate_load_vehicles(buffer_route_xml_file).tolist()
    return output_configuration_file, load_vehicle_list


def generate_load_vehicles(route_file):
    route_tree = ET.parse(route_file)
    root = route_tree.getroot()

    loaded_vehicle_array = np.zeros(3600)
    for subroot in root:
        departure_time = subroot.attrib["depart"]
        departure_time = int(np.floor(float(departure_time)))
        if departure_time >= 3600:
            continue
        loaded_vehicle_array[departure_time] += 1
    return loaded_vehicle_array


def delete_buffer_file(sumo_seed):
    """
    delete the generated configuration file and route file...
    :param sumo_seed:
    :return:
    """
    output_configuration_file = config.buffer_configuration_file + str(sumo_seed) + ".sumocfg"
    route_xml_file = config.buffer_route_head + str(sumo_seed) + ".xml"
    buffer_route_xml_file = os.path.join(config.sumo_configuration_buffer_folder, route_xml_file)

    if os.path.exists(output_configuration_file):
        os.remove(output_configuration_file)
    if os.path.exists(buffer_route_xml_file):
        os.remove(buffer_route_xml_file)


if __name__ == '__main__':
    generate_new_route_configuration()
