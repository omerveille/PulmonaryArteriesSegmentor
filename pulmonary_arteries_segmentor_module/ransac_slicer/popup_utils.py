import math
import time
from typing import Union
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


class ProgressBarTimer:
    """
    Class to wrap a loop in order to compute wait time and percentage of job done.
    """

    def __init__(self, total: int):
        self.total = total
        self.count = 0

    def __enter__(self):
        self.start_time = time.time()
        self.last_second = self.start_time
        self.last_percent = 0

        return self

    def __exit__(self, type, value, traceback):
        pass

    @classmethod
    def format_time(cls, seconds: int) -> str:
        """
        Format second duration into string.
        """
        return str(timedelta(seconds=int(seconds)))

    def update(self) -> tuple[float, float, float, bool]:
        """
        Compute the elapsed, remaining and percentage done.

        Returns
        ----------

        Elasped time, Remaining time, Percentage done and a boolean to suggest an UI update.

        Times returned are expressed in seconds.
        """
        self.count += 1

        current_time = time.time()

        elapsed_time = current_time - self.start_time
        remaining_time = (
            (self.total - self.count) * (elapsed_time / self.count)
            if self.count > 0
            else 0
        )
        percent_done = int(math.floor((self.count / self.total) * 100))

        should_update_ui = False
        if (percent_done - self.last_percent) >= 1:
            should_update_ui = True
            self.last_percent = percent_done

        if (current_time - self.last_second) >= 1:
            should_update_ui = True
            self.last_second = current_time

        return elapsed_time, remaining_time, percent_done, should_update_ui


class CustomProgressBar(ProgressBarTimer):
    """
    Class to ease the creation of a progress bar.
    """

    def __init__(
        self,
        total: int,
        quantity_to_measure: str,
        windowTitle: str = "windowTitle",
        width: Union[int, None] = None,
        height: Union[int, None] = None,
    ):
        super().__init__(total)

        self.quantity_to_measure = quantity_to_measure
        self.elapsed_time = 0
        self.remaining_time = 0

        self.progress_bar = make_custom_progress_bar(
            windowTitle=windowTitle,
            labelText=self.__make_progress_bar_text(),
            width=width,
            height=height,
        )

    def __exit__(self, type, value, traceback):
        self.progress_bar.hide()
        self.progress_bar.close()

    def __make_progress_bar_text(self) -> str:
        return f"{self.count}/{self.total} {self.quantity_to_measure}\nElapsed time: {ProgressBarTimer.format_time(self.elapsed_time)}\nRemaining time: {ProgressBarTimer.format_time(self.remaining_time)}"

    def update(self):
        (
            self.elapsed_time,
            self.remaining_time,
            percent_done,
            should_update_ui,
        ) = super().update()
        if should_update_ui:
            self.progress_bar.labelText = self.__make_progress_bar_text()
            self.progress_bar.value = percent_done
            slicer.app.processEvents()
