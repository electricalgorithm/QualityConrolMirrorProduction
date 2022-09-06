import cv2
import argparse
from src.libs.Debug import Debug
from src.libs.AngleDetection import AngleDetection
from src.libs.RectangularDetection import RectangularDetection


def init_argument_parser():
    """This function handles the flagging system.
    Returns:
        dict: Includes the flag responses of the program.
    """
    parser = argparse.ArgumentParser(
        description="This function gives you the angles of the lines in particular environment.")
    parser.add_argument("-i", "--image_location", nargs="+",
                        required=True, help="Provide your image to calculate angles.")
    parser.add_argument("-t", "--tiles", nargs="+",
                        required=True, help="Provide the tile count for dividing the image into sub-images.")
    parser.add_argument("-lf", "--log_file", nargs="+",
                        required=True, help="The file location to save logs.")
    parser.add_argument("-sd", "--save_dir", nargs="+",
                        required=True, help="The directory location to save step images.")

    return parser.parse_args()


if __name__ == "__main__":
    args = init_argument_parser()

    # Check the tile count.
    if len(args["tiles"]) != 2:
        raise RuntimeError("You need to provide two integer values for tile structure.")

    # Create a debugger to have information messages.
    debugger = Debug(args["log_file"], args["save_dir"])
    debugger.set_level("INFO")

    # Open the image with OpenCV.
    Image = cv2.imread(args["image_location"])

    # Detect the window.
    object_detector = RectangularDetection(Image, debug=debugger)
    object_detector.run()
    ObjectImage = object_detector.get_result()

    # Detect the angles in the ObjectImage.
    angle_detector = AngleDetection(ObjectImage, (int(args["tiles"][0]), int(args["tiles"][1])), debug=debugger)
    angle_detector.run()
    print(f"Results: {angle_detector.get_angles()}")
