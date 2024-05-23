import math
import time
from typing import Union
from collections.abc import Iterable
import slicer
import qt
from datetime import timedelta


def make_custom_progress_bar(
    labelText: str = "labelText",
    windowTitle: str = "windowTitle",
    width: Union[int, None] = None,
    height: Union[int, None] = None,
):
    """
    Wrapper of slicer progress dialog function.
    """
    progress_bar = slicer.util.createProgressDialog(
        parent=slicer.util.mainWindow(),
        autoClose=False,
        labelText=labelText,
        windowTitle=windowTitle,
        value=0,
    )
    shape = (
        progress_bar.width if width is None else width,
        progress_bar.height if height is None else height,
    )
    # Ensure the dialog is deleted when closed
    progress_bar.setAttribute(qt.Qt.WA_DeleteOnClose)
    progress_bar.setCancelButton(None)
    progress_bar.resize(*shape)
    progress_bar.show()
    slicer.app.processEvents()
    return progress_bar


class CustomStatusDialog:
    """
    Class to ease the creation and usage of dialog box containing text.
    """

    def __init__(
        self,
        windowTitle: str = "windowTitle",
        text: str = "text",
        width: Union[int, None] = None,
        height: Union[int, None] = None,
    ):
        dialog = qt.QDialog(slicer.util.mainWindow())
        dialog.setWindowTitle(windowTitle)
        # Ensure the dialog is deleted when closed
        dialog.setAttribute(qt.Qt.WA_DeleteOnClose)
        shape = (
            dialog.width if width is None else width,
            dialog.height if height is None else height,
        )
        dialog.resize(*shape)
        layout = qt.QVBoxLayout()
        label = qt.QLabel(text)
        font = label.font
        font.setPointSize(14)
        label.setFont(font)
        # Center the text
        label.setAlignment(qt.Qt.AlignCenter)
        layout.addWidget(label)
        dialog.setLayout(layout)

        dialog.show()
        slicer.app.processEvents()

        self.label = label
        self.dialog = dialog

    def setText(self, text: str):
        """
        Change text and update UI.
        """
        self.label.setText(text)
        slicer.app.processEvents()

    def close(self):
        self.dialog.close()


class CustomProgressBar:
    """
    Class to ease the creation of a progress bar which display remaining time, elapsed time, and percentage of work done.
    """

    def __init__(
        self,
        iterable: Iterable,
        quantity_to_measure: str,
        windowTitle: str = "windowTitle",
        width: Union[int, None] = None,
        height: Union[int, None] = None,
    ):
        self.total = len(iterable)
        self.iterable = iter(iterable)
        self.count = 0

        self.quantity_to_measure = quantity_to_measure
        self.elapsed_time = 0
        self.remaining_time = 0

        self.progress_bar = make_custom_progress_bar(
            windowTitle=windowTitle,
            labelText=self.__make_progress_bar_text(),
            width=width,
            height=height,
        )

    def __close_bar(self):
        self.progress_bar.hide()
        self.progress_bar.close()

    def __make_progress_bar_text(self) -> str:
        return f"{self.count}/{self.total} {self.quantity_to_measure}\nElapsed time: {self.__format_time(self.elapsed_time)}\nRemaining time: {self.__format_time(self.remaining_time)}"

    def __format_time(self, seconds: int) -> str:
        """
        Format second duration into string.
        """
        return str(timedelta(seconds=int(seconds)))

    def __update(self) -> bool:
        """
        Compute the elapsed, remaining and percentage of work done.

        Returns
        ----------

        A boolean to suggest an UI update.
        """
        self.count += 1

        current_time = time.time()

        self.elapsed_time = current_time - self.start_time
        self.remaining_time = (
            (self.total - self.count) * (self.elapsed_time / self.count)
            if self.count > 0
            else 0
        )
        current_percent_work_done = int(math.floor((self.count / self.total) * 100))

        should_update_ui = False
        if (current_percent_work_done - self.percent_work_done) >= 1:
            should_update_ui = True
            self.percent_work_done = current_percent_work_done

        if (current_time - self.last_second_elapsed) >= 1:
            should_update_ui = True
            self.last_second_elapsed = current_time

        return should_update_ui

    def __iter__(self):
        self.start_time = time.time()
        self.last_second_elapsed = self.start_time
        self.percent_work_done = 0

        try:
            for obj in self.iterable:
                if self.__update():
                    self.progress_bar.labelText = self.__make_progress_bar_text()
                    self.progress_bar.value = self.percent_work_done
                    slicer.app.processEvents()
                yield obj

        finally:
            self.__close_bar()
