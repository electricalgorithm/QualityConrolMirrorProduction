import math
import cv2
import numpy
from libs.Debug import Debug


class AngleDetection:
    """
    This class helps you to find angles in an image.
    """

    def __init__(self, image_file: numpy.ndarray,
                 tiles: tuple,
                 debug: Debug):

        self.debug = debug
        # Check the tiles parameter.
        if tiles is None:
            self.debug.error("You must provide tiles size for griding the image.")

        # Properties to save things later.
        self.image_uploaded = image_file
        self.mini_images = []
        self.rgb_mini_images = []
        self.angles = []
        self.tiles_wanted = tiles
        self.is_algorithm_finished = False

    """
    #################################################
                INTERFACE FOR END-USERS
    #################################################
    """
    def run(self) -> None:
        """
        The function runs the algorithm and returns the resulting angles.
        :return: Angles
        """
        # Divide the images.
        self.mini_images = \
            self.divide_the_image(self.image_uploaded,
                                  tiles=(self.tiles_wanted if self.tiles_wanted is not None else (5, 2)))

        # Travel all the images.
        for _index in range(0, len(self.mini_images)):
            # Use Canny Algorithm to find edges.
            canny_image = self.apply_canny_detection(self.mini_images[_index])
            self.debug.info(f'Canny detection applied to the {_index}th mini-image.')
            self.save_image(f"AD-CannyImage-{_index}", canny_image)

            # Crop the sub-image to remove outer lines.
            cropped_image = self.crop_image_by_percentage(canny_image, 10)
            self.debug.info(f'Cropping the mini-image {_index} had completed.')
            self.save_image(f'AD-CropImage-{_index}', cropped_image)

            # Create a RGB image to draw red Hough Lines onto.
            mini_image_with_color = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2RGB)
            self.rgb_mini_images.append(mini_image_with_color)

            # Find the average angle with Hough transformation method.
            current_angle = self.apply_adaptive_hough_lines(mini_image_with_color, cropped_image)
            self.debug.info(f'Hough transformation applied to the {_index}th mini-image.')
            self.save_image(f'AD-HoughLine-{_index}', mini_image_with_color)

            # Save the result into logs and angles array.
            self.debug.result(f'Average angle for the {_index}th mini-image is {current_angle}.')
            self.angles.append(current_angle)

        # Finish the algorithm.
        self.is_algorithm_finished = True

    def get_combined_image(self) -> numpy.ndarray:
        """
        Returns the combined image.
        :return: Combined sub-images as numpy.ndarray
        """
        return self.create_combine_image()

    def get_angles(self) -> list:
        """
        This function returns the result of the algorithm.
        :return: Angles as list.
        """
        return self.angles if self.is_algorithm_finished else None

    """
    #################################################
                ADDITIONAL FUNCTIONS
    #################################################
    """
    @staticmethod
    def crop_image_by_percentage(image: numpy.ndarray, percentage: int) -> numpy.ndarray:
        """
        Function crops the given image from each border and returns it.
        :param image: The image that ill be cropped.
        :param percentage: Percentage as integer value.
        :return: Cropped image as numpy.ndarray.
        """
        x_size, y_size = image.shape
        x_size_crop_delete = int(x_size * percentage / 100)
        y_size_crop_delete = int(y_size * percentage / 100)
        return image[x_size_crop_delete:-x_size_crop_delete,
                     y_size_crop_delete:-y_size_crop_delete]

    def create_combine_image(self):
        """
        This function combines all the sub-images into one and returns it.
        :return: Combined image as numpy.ndarray
        """
        # Create a class to remove direct access to the image array
        # when using cv2.reshape.
        class ImageToOrder:
            def __init__(self, image_as_numpy_array):
                self._numpy_array_ = image_as_numpy_array

            def get_image(self):
                return self._numpy_array_

        # Convert each object to ImageToOrder object.
        image_to_order_list = []
        for image in self.rgb_mini_images:
            image_object = ImageToOrder(image)
            image_to_order_list.append(image_object)

        # Combine the image.
        image_ordered_list = numpy.reshape(image_to_order_list, self.tiles_wanted)
        col_concat = []
        for list_images_in_a_row in image_ordered_list:
            row_images = [element_per_row.get_image() for element_per_row in list_images_in_a_row]
            row_concat = cv2.hconcat(row_images)
            col_concat.append(row_concat)
        return cv2.vconcat(col_concat)

    def save_image(self, image_name, image_ndarray):
        """
        This function saves the given ndarray as an image.
        :param image_name:
        :param image_ndarray:
        :return:
        """
        if self.debug.current_debug_level >= self.debug.debug_levels.INFO:
            cv2.imwrite(f"{self.debug.image_save_directory}/{image_name}.png", image_ndarray)

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

    """
    #################################################
                IMAGE PROCESSING METHODS
    #################################################
    """
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
            current_angle, lines_count = self._apply_hough_lines_(image_to_put, image, threshold=int(threshold_to_test))
            threshold_to_test *= threshold_divider_constant

            if threshold_to_test <= 0:
                self.debug.error("The Image is not proper for finding its lines.")
        self.debug.info("apply_adaptive_hough_lines(): Function ended.")
        return current_angle

    def _apply_hough_lines_(self, image_to_put: numpy.ndarray, image: numpy.ndarray, threshold=35) -> list:
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

                self.debug.info(f'{i}: rho: {rho}\t'
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

        self.debug.info(f"[{angle_to_return}, {lines_count}]")
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
