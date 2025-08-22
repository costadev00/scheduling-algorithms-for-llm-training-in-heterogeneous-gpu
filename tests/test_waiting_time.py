import pytest
import sys
import pathlib

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR / "heft"))
sys.path.append(str(BASE_DIR / "peft"))

from heft.heft import _compute_waiting_time as heft_waiting, ScheduleEvent as HeftEvent
from peft.peft import _compute_waiting_time as peft_waiting, ScheduleEvent as PeftEvent


def test_waiting_time_heft():
    proc_schedules = {
        0: [HeftEvent(task=0, start=0, end=5, proc=0),
            HeftEvent(task=1, start=5, end=8, proc=0)],
        1: [HeftEvent(task=2, start=2, end=6, proc=1)],
    }
    expected = (0 + 5 + 2) / 3
    assert heft_waiting(proc_schedules) == pytest.approx(expected)


def test_waiting_time_peft():
    proc_schedules = {
        0: [PeftEvent(task=0, start=0, end=5, proc=0),
            PeftEvent(task=1, start=5, end=8, proc=0)],
        1: [PeftEvent(task=2, start=2, end=6, proc=1)],
    }
    expected = (0 + 5 + 2) / 3
    assert peft_waiting(proc_schedules) == pytest.approx(expected)
