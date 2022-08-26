import cv2
import numpy
import math


class Debug:
    """
    This class implements a debugging feature to create log files and
    output informative lines into the console.
    """
    def __init__(self, file_location: str):
        try:
            self.log_file = open(file_location, "a", encoding="utf-8")
        except FileNotFoundError:
            raise Exception("Logs file couldn't be opened.")

        # Create a debug level.
        class DebugLevels:
            INFO = 0
            ERROR = 1
            RESULT = 2

        self.debug_level = DebugLevels()
        self.debug_level_names = ["INFO", "ERROR", "RESULT"]

    def _save_and_return_message_(self, debug_level: int, message: str) -> str:
        """
        This function saves a log into the log file and outputs to console.
        :param debug_level: A debug level indicator as integer.
        :param message:
        :return: log as string
        """
        log = f"[{self.debug_level_names[debug_level]}] {message}"
        self.log_file.write(log + "\n")
        print(log)
        return log

    def info(self, message: str) -> str:
        """
        Creates a information output.
        :param message: The informative message description.
        :return: Log text as string.
        """
        return self._save_and_return_message_(self.debug_level.INFO, message)

    def error(self, message: str) -> str:
        """
        Creates a error-level output.
        :param message: The informative message description.
        :return: Log text as string.
        """
        return self._save_and_return_message_(self.debug_level.ERROR, message)

    def result(self, message: str) -> str:
        """
        Creates a result output.
        :param message: The informative message description.
        :return: Log text as string.
        """
        return self._save_and_return_message_(self.debug_level.RESULT, message)


class AngleDetection:
    """
    This class helps you to find angles in an image.
    """

    def __init__(self, image_file: numpy.ndarray, tiles: tuple, log_file_location: str = "logs.txt"):
        self.debug = Debug(log_file_location)
        # Check the tiles parameter.
        if tiles is None:
            self.debug.error("You must provide tiles size for griding the image.")

        # Properties to save things later.
        self.image_uploaded = image_file
        self.mini_images = []
        self.angles = []
        self.tiles_wanted = tiles
        self.is_algorithm_finished = False

    def run(self) -> None:
        """
        The function runs the algorithm and returns the resulting angles.
        :return: Angles
        """
        # Divide the images.
        self.mini_images = self.divide_the_image(self.image_uploaded,
                                                 tiles=(self.tiles_wanted if self.tiles_wanted is not None else (5, 2)))

        # Travel all the images.
        for _index in range(0, len(self.mini_images)):
            # Make the image binary with Otsu's Method.
            # binary_image = self.apply_binarization(self.mini_images[_index])
            # self.debug.info("Binarization applied to the {_index}th mini-image.")

            # Use Canny's Algorithm to find edges.
            cannied_image = self.apply_canny_detection(self.mini_images[_index])
            self.debug.info(f'Canny detection applied to the {_index}th mini-image.')

            # Find the average angle with Hough transformation method.
            current_angle = self.apply_adaptive_hough_lines(self.mini_images[_index], cannied_image)
            self.debug.info(f'Hough transformation applied to the {_index}th mini-image.')
            self.debug.result(f'Average angle for the {_index}th mini-image is {current_angle}.')
            self.angles.append(current_angle)

            self.is_algorithm_finished = True

    def get_angles(self) -> list:
        """
        This function returns the result of the algorithm.
        :return: Angles as list.
        """
        return self.angles if self.is_algorithm_finished else None

    def apply_adaptive_hough_lines(self, image_to_put: numpy.ndarray, image: numpy.ndarray) -> float:
        """
        A function find Hough Lines for adaptive thresholds.
        :param image_to_put: Image to put lines as numpy.ndarray.
        :param image: Image to find lines as numpy.ndarry.
        :return: A float angle.
        """
        self.debug.info("apply_adaptive_hough_lines(): Function started.")
        threshold_to_test = 150
        threshold_divider_constant = 0.95
        current_angle = 0.0

        lines_count = 0
        while lines_count < 2:
            current_angle, lines_count = self.apply_hough_lines(image_to_put, image, threshold=int(threshold_to_test))
            threshold_to_test *= threshold_divider_constant

            if threshold_to_test <= 0:
                self.debug.error("The Image is not proper for finding its lines.")
        self.debug.info("apply_adaptive_hough_lines(): Function ended.")
        return current_angle

    def apply_hough_lines(self, image_to_put: numpy.ndarray, image: numpy.ndarray, threshold=35) -> list:
        """
        This method firstly finds the border lines using Hough's transformation method with experimentally
        predetermined threshold. Afterwards, it marks the original image with that border line.
        :param image_to_put: The 3-channel image that will be used for showing lines.
        :param image: The image that will be looked for lines.
        :param threshold: Default is 90.
        :return: The image with lines as numpy.ndarray.
        """
        self.debug.info("apply_hough_lines(): Function started.")
        sum_of_all_theta = 0
        angle_to_return = 0
        lines_count = 0

        # Find the line.
        lines = cv2.HoughLines(image, 1, numpy.pi / 180, threshold, None, 0, 0)

        # Mark the line into the original image.
        if lines is not None:
            self.debug.info("---------APPLY_HOUGH_LINES---------")
            for i in range(0, len(lines)):
                # Get the angle and the starting position.
                rho = lines[i][0][0]
                theta = lines[i][0][1]

                # Create the red line and insert it to the image_to_put.
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
                cv2.line(image_to_put, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)

                if rho < 0:
                    new_theta = math.pi - theta
                else:
                    new_theta = theta

                self.debug.result(f'{i}: rho: {rho}\t'
                                  f'theta: {int(math.degrees(theta))}\t'
                                  f'new_theta: {int(math.degrees(new_theta))}')
                sum_of_all_theta += math.degrees(new_theta)

            # Find the average angle.
            try:
                average_angle = sum_of_all_theta / len(lines)
            except TypeError as error:
                self.debug.error(f"{error}")
                average_angle = 0

            # Find the lines count.
            lines_count = len(lines) if lines is not None else 0

            # Angle correction due to OpenCV lib.
            if 90 - average_angle < 0:
                angle_to_return = (90 - average_angle) * -1
            else:
                angle_to_return = 90 - average_angle

        self.debug.result(f"[{angle_to_return}, {lines_count}]")
        self.debug.info("apply_hough_lines(): Function ended.")
        return [angle_to_return, lines_count]

    def apply_canny_detection(self, image: numpy.ndarray, min_threshold=100, max_threshold=250) -> numpy.ndarray:
        """
        This function applies the Canny algorithm to distinguish a object border.
        :param image: A binary image.
        :param min_threshold: Default is 100.
        :param max_threshold: Default is 250.
        :return: A numpy.ndarray that has only borders of the image.
        """
        self.debug.info("apply_canny_detection(): Function started.")
        image_to_return = cv2.Canny(image, min_threshold, max_threshold)
        self.debug.info("apply_canny_detection(): Function ended.")
        return image_to_return

    def apply_binarization(self, image: numpy.ndarray) -> numpy.ndarray:
        """
        The function gets an image input, and apply Otsu's Method
        to find its binary representation.
        :param image: Original 3-channel image to perform Otsu's method.
        :return: The binary image as a numpy.ndarray.
        """
        self.debug.info("apply_binarization(): Function started.")
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_image = cv2.bitwise_not(binary_image)
        return binary_image

    def divide_the_image(self, image_to_divide: numpy.ndarray, tiles: tuple = (5, 2)) -> list:
        """
        This function breaks the image into row and column
        count given in the tiles parameter.
        :param image_to_divide: Image numpy.ndarray to divide into tiles.
        :param tiles: (row_count, column_count)
        :return: A list contains the image parts.
        """
        self.debug.info("divide_the_image(): Function started.")
        rows_size = image_to_divide.shape[0] // tiles[0]
        cols_size = image_to_divide.shape[1] // tiles[1]

        # Crop the images and put it into mini_images list.
        mini_images = []
        for start_index_col in range(0, image_to_divide.shape[1], cols_size):
            for start_index_row in range(0, image_to_divide.shape[0], rows_size):
                cropped_part = image_to_divide[
                               start_index_row:(start_index_row + rows_size),
                               start_index_col:(start_index_col + cols_size)
                               ]
                # Ignore any small parts which is redundant.
                if cropped_part.shape[0] == rows_size and cropped_part.shape[1] == cols_size:
                    mini_images.append(cropped_part)

        self.debug.info("Image division is completed.")
        self.debug.info("divide_the_image(): Function ended.")
        return mini_images

    def save_images(self, images: list, prefix="mini-", file_format="png") -> None:
        """
        This function saves the cropped images into the root directory.
        :param images: Image list.
        :param prefix: Defaults "mini-".
        :param file_format: Defaults "png"
        :return: None
        """
        self.debug.info("save_images(): Function started.")
        _index = 0
        for image in images:
            cv2.imwrite(f'saves/{prefix}{_index}.{file_format}', image)
            _index += 1
        self.debug.info("Divided image parts are saved into 'saves/'.")
        self.debug.info("save_images(): Function ended.")


class RectangularDetection:
    """
    This class implements the main algorithm which finds the rectangle shaped
    object in a photo and return its top view,
    """

    def __init__(self, image_file: numpy.ndarray, log_file_location: str = "logs.txt"):
        self.image_uploaded = image_file
        self.debug = Debug(log_file_location)

        self.steps = []
        self.corners = []
        self.end_image = None
        self.is_algorithm_finished = False

    def run(self) -> None:
        """
        It runs the algorithm.
        :return: None
        """
        # Binarization
        self.end_image = self.apply_binarization(self.image_uploaded)
        self.steps.append(self.end_image)

        # Closing
        self.end_image = self.apply_morphological_operations(self.end_image)
        self.steps.append(self.end_image)

        # Corner Finding
        corners_by_contour = self.apply_contour_find(self.end_image)
        corners_by_harris = self.apply_harris_corner_detection(self.end_image)
        self.corners = self.validate_corners(corners_by_harris, corners_by_contour)
        self.corners = self.order_corners(self.corners)

        # Transformation
        self.end_image = self.apply_warp_transformation(self.image_uploaded, self.corners, (800, 600))
        self.steps.append(self.end_image)

        # Binarization
        self.end_image = self.apply_binarization(self.end_image)
        self.steps.append(self.end_image)

        # Remove borders
        self.end_image = self.remove_borders(self.end_image)
        self.steps.append(self.end_image)

        # Finish the algorithm
        self.is_algorithm_finished = True
        self.save_all_steps()

    def get_result(self) -> numpy.ndarray:
        """
        This function returns the resulting image.
        :return: A image contains only a rectangular object as numpy.ndarray.
        """
        return self.end_image if self.is_algorithm_finished else None

    def save_image(self, image: numpy.ndarray, file_location: str) -> None:
        """
        Saves the given numpy.ndarray as a image to the file system.
        :param image: Image as numpy.ndarray
        :param file_location: A file location to save image as string.
        :return: None
        """
        self.debug.info("save_image(): Function started.")
        cv2.imwrite(file_location, image)
        self.debug.result(f"Image is saved to {file_location}.")
        self.debug.info("save_image(): Function ended.")

    def save_all_steps(self) -> None:
        """
        The function saves all the steps as PNG.
        :return: None
        """
        step_index = 1
        for step in self.steps:
            self.save_image(step, f"step-{step_index}.png")
            step_index += 1

    def apply_binarization(self, image: numpy.ndarray) -> numpy.ndarray:
        """
        The function gets an image input, and apply Otsu's Method to find its binary representation.
        :param image: Original 3-channel image to perform Otsu's method.
        :return: The binary image as a numpy.ndarray.
        """
        self.debug.info("make_binary_image(): Function started.")
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_image = cv2.bitwise_not(binary_image)
        self.debug.info("make_binary_image(): Function ended.")
        return binary_image

    def apply_morphological_operations(self, image: numpy.ndarray, kernel_size: int = 25):
        """
        The function applies closing into the image with very-big kernel sizes.
        :param image: Binary image to perform method.
        :param kernel_size: Defaults to 25.
        :return: A binary image as numpy.ndarray.
        """
        self.debug.info("apply_morphological_operations(): Function started.")
        element = cv2.getStructuringElement(cv2.MORPH_RECT,
                                            (2 * kernel_size + 1, 2 * kernel_size + 1),
                                            (kernel_size, kernel_size))
        image_to_return = cv2.erode(image, element)
        image_to_return = cv2.dilate(image_to_return, element)
        self.debug.info("apply_morphological_operations(): Function ended.")
        return image_to_return

    def apply_gaussian_filter(self, image: numpy.ndarray, window_size=51, gaussian_sigma=5) -> numpy.ndarray:
        """
        Function returns the cv2 Image instance that has Gaussian blur with predefined window size and gaussian
        sigma ratio.
        :param image: Original image to perform blur onto.
        :param window_size: Default is 5.
        :param gaussian_sigma: Default is 5.
        :return: The blurred image as a numpy.ndarray.
        """
        self.debug.info("blurred_image(): Function started.")
        blurred_image = cv2.GaussianBlur(image, (window_size, window_size), gaussian_sigma)
        self.debug.info("blurred_image(): Function ended.")
        return blurred_image

    def apply_contour_find(self, image: numpy.ndarray):
        """
        This method firstly finds the contour of each object in the image, select the biggest one, and try to
        find a polygon within the contour. It returns the corners of the polygon.
        :param image: Binary image to search for contour.
        :return: A list that includes corner coordinates for the biggest blob.
        """
        self.debug.info("apply_contour_find(): Function started.")
        _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.debug.info(f"{len(contours)} counters found. Selecting the biggest one.")

        contour_wanted = contours[0]
        for contour in contours:
            if cv2.contourArea(contour_wanted) <= cv2.contourArea(contour):
                contour_wanted = contour
        self.debug.info("The biggest contour is selected.")

        epsilon = 0.01 * cv2.arcLength(contour_wanted, True)
        approx = cv2.approxPolyDP(contour_wanted, epsilon, True)
        self.debug.info("Approximated the polygon, returning the corners.")

        self.debug.info("apply_contour_find(): Function ended.")
        return approx

    def apply_harris_corner_detection(self, image: numpy.ndarray):
        """
        This function applies the Harris Corner detection algorithm and removes the redundant near-by pixels.
        :param image: The binary image to look for corners.
        :return: A list that includes corner coordinates.
        """
        self.debug.info("apply_harris_corner_detection(): Function started.")
        corners = []

        dst = cv2.cornerHarris(image, 10, 7, 0.04)
        self.debug.info("Corner Harris method is applied, and corners are detected.")
        thresh = 0.25 * dst.max()
        self.debug.info(f"Threshold for corner extraction is {thresh}. Selection is starting.")
        for j in range(0, dst.shape[0]):
            for i in range(0, dst.shape[1]):
                if dst[j, i] > thresh:
                    corners.append((i, j))
                    self.debug.result(f"Corner is found and added to list: ({i}, {j})")

        self.debug.info("Algorithm started to delete nearby corners.")
        minimum_corners = []
        if len(corners) > 4:
            for corner in corners:
                if corner not in minimum_corners:
                    corner_x = corner[0]
                    corner_y = corner[1]

                    # Check if there is similar pixel in minimum array.
                    is_there_similar_pixel = False
                    for x_value in range(corner_x - 15, corner_x + 15):
                        for y_value in range(corner_y - 15, corner_y + 15):
                            if (x_value, y_value) in minimum_corners:
                                is_there_similar_pixel = True

                    if not is_there_similar_pixel:
                        minimum_corners.append(corner)

        self.debug.info("apply_harris_corner_detection(): Function ended.")
        return minimum_corners

    # @TODO: Implement this method and return the average corners.
    def validate_corners(self, corners_by_harris: list, corners_by_contour: list) -> list:
        self.debug.info("validate_corners(): Function started.")
        self.debug.error("NOT IMPLEMENTED!")
        self.debug.info("validate_corners(): Function ended.")
        return corners_by_harris

    def order_corners(self, corners: list) -> list:
        """
        This function gets four corners coordinates, and orders it from left to right, top to bottom.
        :param corners: A list contains four corners.
        :return: [left_top, right_top, left_bottom, right_bottom]
        """
        self.debug.info("order_corners(): Function started.")
        corners.sort(key=lambda corner: corner[0] + corner[1])
        sorted_coordinates_by_location = [corners[0], corners[2],
                                          corners[1], corners[3]]
        self.debug.info("order_corners(): Function ended.")
        return sorted_coordinates_by_location

    def apply_warp_transformation(self, image: numpy.ndarray, corners: list, new_image_size: tuple = (1000, 500)) \
            -> numpy.ndarray:
        """
        This function applies Warp transformation to get right perspective for the object that is
        described with its corners.
        :param image: A image to apply transformation.
        :param corners: A list of corners of the object to get from the image.
        :param new_image_size: A tuple of size data for the new image.
        :return: A image as numpy.ndarray
        """
        self.debug.info("apply_warp_transformation(): Function started.")
        top_left_corner_new = (0, 0)
        top_right_corner_new = (new_image_size[0] - 1, 0)
        bottom_left_corner_new = (0, new_image_size[1] - 1)
        bottom_right_corner_new = (new_image_size[0] - 1, new_image_size[1] - 1)
        new_corners = numpy.float32([top_left_corner_new, top_right_corner_new,
                                     bottom_left_corner_new, bottom_right_corner_new])
        old_corners = numpy.float32(corners)
        perspective = cv2.getPerspectiveTransform(old_corners, new_corners)
        resulting_object_image = cv2.warpPerspective(image, perspective, new_image_size,
                                                     flags=cv2.INTER_LINEAR)
        self.debug.info("apply_warp_transformation(): Function ended.")
        return resulting_object_image

    def remove_borders(self, image: numpy.ndarray) -> numpy.ndarray:
        """
        This function removes the borders of the rectangular window object.
        :param image:  A image as numpy.ndarray
        :return: Resulting image as numpy.ndarray
        """
        self.debug.info("remove_borders(): Function started.")

        height, width = image.shape
        first_white_from_left = 0
        first_white_from_right = 0
        threshold_to_be_white_line = 180

        self.debug.info(f"Starting the remove left side. Threshold: {threshold_to_be_white_line}")
        for pixel_index_from_left in range(0, width):
            if numpy.mean(image[:, pixel_index_from_left]) > threshold_to_be_white_line:
                first_white_from_left = pixel_index_from_left
                break

        self.debug.info(f"Starting the remove right side. Threshold: {threshold_to_be_white_line}")
        for pixel_index_from_right in range(width - 1, 0, -1):
            if numpy.mean(image[:, pixel_index_from_right]) > threshold_to_be_white_line:
                first_white_from_right = pixel_index_from_right
                break

        first_biggest_white_from_top = 0
        first_biggest_white_from_bottom = 0
        threshold_to_be_white_lines = 240

        self.debug.info(f"Starting the remove top side. Threshold: {threshold_to_be_white_lines}")
        for pixel_index_from_top in range(0, height):
            if numpy.mean(image[pixel_index_from_top:(pixel_index_from_top + 5), :]) \
                    > threshold_to_be_white_lines:
                first_biggest_white_from_top = pixel_index_from_top
                break

        self.debug.info(f"Starting the remove bottom side. Threshold: {threshold_to_be_white_lines}")
        for pixel_index_from_bottom in range(height - 1, 6, -1):
            if numpy.mean(image[(pixel_index_from_bottom - 5):pixel_index_from_bottom, :]) \
                    > threshold_to_be_white_lines:
                first_biggest_white_from_bottom = pixel_index_from_bottom - 5
                break

        self.debug.result(f"Image is cropped from {first_biggest_white_from_top} to {first_biggest_white_from_bottom}"
                          f" on vertical, and from {first_white_from_left} to {(first_white_from_right + 1)} on "
                          f"horizontal.")
        image_cropped = image[first_biggest_white_from_top:first_biggest_white_from_bottom,
                              first_white_from_left:(first_white_from_right + 1)]
        self.debug.info("remove_borders(): Function ended.")
        return image_cropped


# Main function
if __name__ == "__main__":
    Image = cv2.imread("reference.jpg")

    # Detect the window.
    object_detector = RectangularDetection(Image)
    object_detector.run()
    ObjectImage = object_detector.get_result()

    cv2.imwrite("object_image.png", ObjectImage)
