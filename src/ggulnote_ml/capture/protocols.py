from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from .config import ProtocolsConfig, StaticProtocolConfig, VerticalClickProtocolConfig


@dataclass(frozen=True)
class ProtocolSegment:
    segment_index: int
    repeat_index: int
    target_index: int
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    duration_ms: float
    direction: str
    usable_start_ms: float = -1.0
    usable_end_ms: float = -1.0
    training_candidate: bool = True


@dataclass(frozen=True)
class ProtocolFrameState:
    protocol_id: str
    split: str
    phase: str
    segment_index: Optional[int]
    repeat_index: Optional[int]
    target_index: Optional[int]
    target_x: Optional[float]
    target_y: Optional[float]
    direction: str
    is_settling: bool
    is_usable_window: bool
    is_training_sample: bool
    show_guide_line: bool
    label_latency_ms: float
    confirmation_required: bool = False
    is_confirmed: bool = False
    confirmation_timestamp_ns: Optional[int] = None
    confirmation_offset_ms: Optional[float] = None
    target_elapsed_ms: Optional[float] = None
    fixation_progress: float = 0.0


@dataclass(frozen=True)
class ProtocolPlan:
    protocol_id: str
    split: str
    initial_delay_ms: float
    completion_duration_ms: float
    segments: Tuple[ProtocolSegment, ...]
    settling_duration_ms: float
    usable_duration_ms: float
    show_guide_line: bool
    latency_ms: float = 0.0
    confirmation_required: bool = False
    confirmation_transition_ms: float = 0.0
    simulation_confirm_after_ms: float = 0.0

    @property
    def active_duration_ms(self) -> float:
        return sum(segment.duration_ms for segment in self.segments)

    @property
    def total_duration_ms(self) -> float:
        return self.initial_delay_ms + self.active_duration_ms + self.completion_duration_ms

    def state_at(self, elapsed_ms: float) -> ProtocolFrameState:
        if elapsed_ms < 0:
            raise ValueError("Protocol elapsed time must be non-negative.")
        if elapsed_ms < self.initial_delay_ms:
            return self._empty_state("ready")

        active_elapsed = elapsed_ms - self.initial_delay_ms
        cursor = 0.0
        for segment in self.segments:
            segment_end = cursor + segment.duration_ms
            if active_elapsed < segment_end:
                offset = active_elapsed - cursor
                progress = min(1.0, max(0.0, offset / segment.duration_ms))
                x = segment.start_x + (segment.end_x - segment.start_x) * progress
                y = segment.start_y + (segment.end_y - segment.start_y) * progress
                if segment.usable_start_ms >= 0:
                    settling = offset < segment.usable_start_ms
                    usable = segment.usable_start_ms <= offset < segment.usable_end_ms
                else:
                    settling = offset < self.settling_duration_ms
                    usable = (
                        offset >= self.settling_duration_ms
                        and offset < self.settling_duration_ms + self.usable_duration_ms
                    )
                return ProtocolFrameState(
                    protocol_id=self.protocol_id,
                    split=self.split,
                    phase="target",
                    segment_index=segment.segment_index,
                    repeat_index=segment.repeat_index,
                    target_index=segment.target_index,
                    target_x=x,
                    target_y=y,
                    direction=segment.direction,
                    is_settling=settling,
                    is_usable_window=usable,
                    is_training_sample=usable and self.split == "train" and segment.training_candidate,
                    show_guide_line=self.show_guide_line,
                    label_latency_ms=self.latency_ms,
                )
            cursor = segment_end

        completion_elapsed = active_elapsed - cursor
        if completion_elapsed < self.completion_duration_ms:
            return self._empty_state("complete_message")
        return self._empty_state("finished")

    def _empty_state(self, phase: str) -> ProtocolFrameState:
        return ProtocolFrameState(
            protocol_id=self.protocol_id,
            split=self.split,
            phase=phase,
            segment_index=None,
            repeat_index=None,
            target_index=None,
            target_x=None,
            target_y=None,
            direction="",
            is_settling=False,
            is_usable_window=False,
            is_training_sample=False,
            show_guide_line=False,
            label_latency_ms=self.latency_ms,
        )


class ProtocolRunner:
    """Evaluate one protocol, including participant-confirmed static targets.

    Timed center plans delegate to :meth:`ProtocolPlan.state_at`.
    Interactive static plans advance only after a valid click/Space confirmation,
    a post-confirmation capture interval, and a short transition. This keeps the
    reusable protocol description separate from GUI input handling.
    """

    def __init__(self, plan: ProtocolPlan) -> None:
        self.plan = plan
        self._segment_index = 0
        self._segment_started_ms = plan.initial_delay_ms
        self._confirmed_at_ms: Optional[float] = None
        self._confirmation_timestamp_ns: Optional[int] = None
        self._completion_started_ms: Optional[float] = None

    def confirm(self, elapsed_ms: float, unix_timestamp_ns: int) -> bool:
        """Confirm the visible target; return False for early or irrelevant input."""

        if unix_timestamp_ns <= 0:
            raise ValueError("Confirmation timestamp must be positive Unix nanoseconds.")
        state = self.state_at(elapsed_ms)
        if (
            not self.plan.confirmation_required
            or state.phase != "target"
            or state.is_confirmed
            or state.target_elapsed_ms is None
            or state.target_elapsed_ms < self.plan.settling_duration_ms
        ):
            return False
        self._confirmed_at_ms = elapsed_ms
        self._confirmation_timestamp_ns = unix_timestamp_ns
        return True

    def state_at(self, elapsed_ms: float) -> ProtocolFrameState:
        if not self.plan.confirmation_required:
            return self.plan.state_at(elapsed_ms)
        if elapsed_ms < 0:
            raise ValueError("Protocol elapsed time must be non-negative.")
        if elapsed_ms < self.plan.initial_delay_ms:
            return self.plan._empty_state("ready")

        self._advance_completed_segments(elapsed_ms)
        if self._segment_index >= len(self.plan.segments):
            if self._completion_started_ms is None:
                self._completion_started_ms = elapsed_ms
            if elapsed_ms - self._completion_started_ms < self.plan.completion_duration_ms:
                return self.plan._empty_state("complete_message")
            return self.plan._empty_state("finished")

        segment = self.plan.segments[self._segment_index]
        target_elapsed_ms = max(0.0, elapsed_ms - self._segment_started_ms)
        confirmed = self._confirmed_at_ms is not None
        confirmation_offset_ms = (
            self._confirmed_at_ms - self._segment_started_ms if confirmed else None
        )
        post_confirmation_ms = (
            max(0.0, elapsed_ms - self._confirmed_at_ms) if confirmed else None
        )
        usable = bool(
            confirmed
            and post_confirmation_ms is not None
            and post_confirmation_ms < self.plan.usable_duration_ms
        )
        settling = not confirmed
        fixation_progress = (
            1.0
            if self.plan.settling_duration_ms == 0
            else min(1.0, target_elapsed_ms / self.plan.settling_duration_ms)
        )
        return ProtocolFrameState(
            protocol_id=self.plan.protocol_id,
            split=self.plan.split,
            phase="target",
            segment_index=segment.segment_index,
            repeat_index=segment.repeat_index,
            target_index=segment.target_index,
            target_x=segment.start_x,
            target_y=segment.start_y,
            direction=segment.direction,
            is_settling=settling,
            is_usable_window=usable,
            is_training_sample=(
                usable and self.plan.split == "train" and segment.training_candidate
            ),
            show_guide_line=False,
            label_latency_ms=self.plan.latency_ms,
            confirmation_required=True,
            is_confirmed=confirmed,
            confirmation_timestamp_ns=self._confirmation_timestamp_ns,
            confirmation_offset_ms=confirmation_offset_ms,
            target_elapsed_ms=target_elapsed_ms,
            fixation_progress=fixation_progress,
        )

    def _advance_completed_segments(self, elapsed_ms: float) -> None:
        while self._confirmed_at_ms is not None:
            segment_end_ms = (
                self._confirmed_at_ms
                + self.plan.usable_duration_ms
                + self.plan.confirmation_transition_ms
            )
            if elapsed_ms < segment_end_ms:
                return
            self._segment_index += 1
            self._segment_started_ms = segment_end_ms
            self._confirmed_at_ms = None
            self._confirmation_timestamp_ns = None
            if self._segment_index >= len(self.plan.segments):
                self._completion_started_ms = segment_end_ms
                return


def _linspace(start: float, stop: float, count: int) -> List[float]:
    if count == 1:
        return [(start + stop) / 2]
    return [start + index * (stop - start) / (count - 1) for index in range(count)]


def build_static_plan(
    config: StaticProtocolConfig,
    initial_delay_ms: float,
    completion_duration_ms: float,
) -> ProtocolPlan:
    positions = [
        (x, y)
        for y in (list(config.y_positions) or _linspace(config.y_min, config.y_max, config.rows))
        for x in _linspace(config.x_min, config.x_max, config.columns)
    ]
    segments = []
    segment_index = 0
    for repeat_index in range(config.repeats):
        order = list(range(len(positions)))
        random.Random(config.random_seed + repeat_index).shuffle(order)
        for target_index in order:
            x, y = positions[target_index]
            segments.append(
                ProtocolSegment(
                    segment_index=segment_index,
                    repeat_index=repeat_index,
                    target_index=target_index,
                    start_x=x,
                    start_y=y,
                    end_x=x,
                    end_y=y,
                    duration_ms=(
                        config.simulation_confirm_after_ms
                        + config.capture_duration_ms
                        + config.transition_duration_ms
                    ),
                    direction="static",
                )
            )
            segment_index += 1
    return ProtocolPlan(
        protocol_id=config.protocol_id,
        split=config.split,
        initial_delay_ms=initial_delay_ms,
        completion_duration_ms=completion_duration_ms,
        segments=tuple(segments),
        settling_duration_ms=config.minimum_fixation_ms,
        usable_duration_ms=config.capture_duration_ms,
        show_guide_line=False,
        confirmation_required=config.confirmation_required,
        confirmation_transition_ms=config.transition_duration_ms,
        simulation_confirm_after_ms=config.simulation_confirm_after_ms,
    )


def build_vertical_click_plan(
    config: VerticalClickProtocolConfig,
    initial_delay_ms: float,
    completion_duration_ms: float,
) -> ProtocolPlan:
    """Build left/center/right click targets, each traversed down and back up."""

    segments = []
    segment_index = 0
    y_positions = _linspace(config.y_min, config.y_max, config.rows)
    for column_index, x in enumerate(config.columns):
        traversals = (
            (0, tuple(enumerate(y_positions)), "top_to_bottom"),
            (1, tuple(reversed(tuple(enumerate(y_positions)))), "bottom_to_top"),
        )
        for repeat_index, indexed_positions, direction in traversals:
            for row_index, y in indexed_positions:
                segments.append(
                    ProtocolSegment(
                        segment_index=segment_index,
                        repeat_index=repeat_index,
                        target_index=column_index * config.rows + row_index,
                        start_x=x,
                        start_y=y,
                        end_x=x,
                        end_y=y,
                        duration_ms=(
                            config.simulation_confirm_after_ms
                            + config.capture_duration_ms
                            + config.transition_duration_ms
                        ),
                        direction=direction,
                    )
                )
                segment_index += 1
    return ProtocolPlan(
        protocol_id=config.protocol_id,
        split=config.split,
        initial_delay_ms=initial_delay_ms,
        completion_duration_ms=completion_duration_ms,
        segments=tuple(segments),
        settling_duration_ms=config.minimum_fixation_ms,
        usable_duration_ms=config.capture_duration_ms,
        show_guide_line=False,
        confirmation_required=config.confirmation_required,
        confirmation_transition_ms=config.transition_duration_ms,
        simulation_confirm_after_ms=config.simulation_confirm_after_ms,
    )


def build_center_plan(protocol_id: str, duration_ms: float) -> ProtocolPlan:
    """Create a non-training center fixation used for baseline and stabilization."""

    segment = ProtocolSegment(
        segment_index=0,
        repeat_index=0,
        target_index=0,
        start_x=0.5,
        start_y=0.5,
        end_x=0.5,
        end_y=0.5,
        duration_ms=duration_ms,
        direction="static",
        usable_start_ms=0.0,
        usable_end_ms=duration_ms,
        training_candidate=False,
    )
    return ProtocolPlan(protocol_id, "reference", 0, 0, (segment,), 0, duration_ms, False)


def build_protocol_plans(config: ProtocolsConfig) -> Tuple[ProtocolPlan, ...]:
    train = build_static_plan(config.train_static, 0, 0)
    transition = build_center_plan("protocol_transition", config.transition_ms)
    vertical_click = build_vertical_click_plan(config.vertical_click, 0, 0)
    refix = build_center_plan("center_refix", config.center_refix_ms)
    evaluation = build_static_plan(config.evaluation_static, 0, config.completion_duration_ms)
    return (
        build_center_plan("intro_center", config.intro_center_ms),
        train,
        transition,
        vertical_click,
        refix,
        evaluation,
    )


def select_protocols(plans: Sequence[ProtocolPlan], names: Iterable[str]) -> Tuple[ProtocolPlan, ...]:
    requested = tuple(names)
    if not requested or requested == ("all",):
        return tuple(plans)
    by_id = {plan.protocol_id: plan for plan in plans}
    missing = [name for name in requested if name not in by_id]
    if missing:
        raise ValueError("Unknown protocol id(s): %s" % ", ".join(missing))
    return tuple(by_id[name] for name in requested)


def normalized_coordinates(x: float, y: float, width: int, height: int) -> Tuple[int, int, float, float]:
    if not 0 <= x <= 1 or not 0 <= y <= 1:
        raise ValueError("Target coordinates must be normalized to [0, 1].")
    return round(x * (width - 1)), round(y * (height - 1)), x - 0.5, y - 0.5


def write_protocol_events(path: Path, plan: ProtocolPlan) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "protocol",
                "split",
                "segment",
                "repeat",
                "target",
                "x",
                "y",
                "direction",
                "confirmation_required",
            ]
        )
        for segment in plan.segments:
            writer.writerow(
                [
                    plan.protocol_id,
                    plan.split,
                    segment.segment_index,
                    segment.repeat_index,
                    segment.target_index,
                    "%.6f" % segment.start_x,
                    "%.6f" % segment.start_y,
                    segment.direction,
                    int(plan.confirmation_required),
                ]
            )
