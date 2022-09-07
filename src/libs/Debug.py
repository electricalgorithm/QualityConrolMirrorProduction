from datetime import date


class Debug:
    """
    This class implements a debugging feature to create log files and
    output informative lines into the console.
    """
    def __init__(self,
                 log_file_location: str,
                 image_save_location: str):
        try:
            self.log_file = open(log_file_location, "a", encoding="utf-8")
        except FileNotFoundError:
            raise Exception("Logs file couldn't be opened.")

        self._add_initial_header_()
        self.image_save_directory = image_save_location

        # Create a debug level.
        class DebugLevels:
            RESULT = 0
            INFO = 1
            ERROR = 2

        self.debug_levels = DebugLevels()
        self.debug_level_names = ["RESULT", "INFO", "ERROR"]
        self.current_debug_level = self.debug_levels.INFO

    def _add_initial_header_(self):
        # Add date into log file.
        today = date.today()
        date_string = today.strftime("%d/%m/%Y")
        self.log_file.write("\n\n=====================================\n")
        self.log_file.write(f"Algorithm started on {date_string}.\n")
        self.log_file.write("=====================================\n")

    def _save_and_return_message_(self, debug_level: int, message: str) -> str:
        """
        This function saves a log into the log file and outputs to console.
        :param debug_level: A debug level indicator as integer.
        :param message:
        :return: log as string
        """
        log = f"[{self.debug_level_names[debug_level]}] {message}"
        self.log_file.write(log + "\n")
        # Print the log if it's debug level satisfy the current level.
        if self.current_debug_level >= debug_level:
            print(log)
        return log

    def set_level(self, level_string: str) -> None:
        """
        This function sets a debug level for outputting to the console.
        :param level_string: "RESULT", "INFO", "ERROR"
        :return: None
        """
        if level_string == "INFO":
            self.current_debug_level = self.debug_levels.INFO
        elif level_string == "RESULT":
            self.current_debug_level = self.debug_levels.RESULT
        elif level_string == "ERROR":
            self.current_debug_level = self.debug_levels.ERROR

    def info(self, message: str) -> str:
        """
        Creates a information output.
        :param message: The informative message description.
        :return: Log text as string.
        """
        return self._save_and_return_message_(self.debug_levels.INFO, message)

    def error(self, message: str) -> str:
        """
        Creates a error-level output.
        :param message: The informative message description.
        :return: Log text as string.
        """
        return self._save_and_return_message_(self.debug_levels.ERROR, message)

    def result(self, message: str) -> str:
        """
        Creates a result output.
        :param message: The informative message description.
        :return: Log text as string.
        """
        return self._save_and_return_message_(self.debug_levels.RESULT, message)