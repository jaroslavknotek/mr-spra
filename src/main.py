import os
import cv2
import json
import argparse
import analyser
import img_logger
import logging

print("Logging set to INFO")
logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser()

parser.add_argument('--input', type=str, required=True, help="path to directory with images")
parser.add_argument('--cfg', type=str, required=True, help="path to config file")
parser.add_argument('--network', type=str, required=True, help="path to network pb file")
parser.add_argument('--output', type=str, required=True, help="path to output directory")
parser.add_argument("--log", type=str, help="increase output verbosity", choices=["SAVE", "PRINT"])
args = parser.parse_args()

path_data = args.input
pictures = [(f, os.path.join(path_data, f)) for f in os.listdir(path_data)
            if os.path.isfile(os.path.join(path_data, f))]

if args.log == "SAVE":
    img_logger.start_logging_to_file("../debug")
    logging.info("setting up debugger to save images")
elif args.log == "PRINT":
    img_logger.start_logging_to_plots()
    logging.info("setting up debugger to plot images")

logging.info("loading config")

with open(args.cfg, "r") as f:
    config = json.load(f)

logging.info("starting processing data")
analysed = {}

for filename, filepath in pictures:
    logging.info("processing {}".format(filename))

    image = cv2.imread(filepath)
    try:
        analysed[filename] = analyser.analyse_image(image, config, args.network)
    except Exception as e:
        logging.error(e)

    logging.info("image processed {}".format(filename))

output_json = os.path.join(args.output, "output.json")

with open(output_json, 'w') as outfile:
    json.dump(analysed, outfile)

logging.info("result json saved to '{}'".format(output_json))
