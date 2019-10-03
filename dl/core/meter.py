from typing import Any, Dict
from collections import defaultdict

from torch import nn


class MetricManager:

    def __init__(self):
        self._meter = AverageMeter()
        self._all_epoch_values: [float] = []
        self._last_batch_value: float = None

    def add_batch_value(self, value: float, batch_size: int):
        # Set correct n to avoid batch size influence on mean value
        self._meter.add(value, n=batch_size)

        self._last_batch_value = value

    def begin_epoch(self):
        self._meter.reset()

    def end_epoch(self):
        self._all_epoch_values.append(self._meter.value())
        self._meter.reset()

    @property
    def last_batch_value(self) -> float:
        return self._last_batch_value

    # @property
    # def current_epoch_value(self) -> float:
    #     return self._meter.value()

    @property
    def last_epoch_value(self) -> float:
        return self._all_epoch_values[-1]

    @property
    def all_epoch_values(self) -> [float]:
        return self._all_epoch_values

    def get_best_epoch_value(self, minimize: bool) -> float:
        if minimize:
            return min(self._all_epoch_values)
        else:
            return max(self._all_epoch_values)


class MetricsManager:

    def __init__(self):
        self._managers: Dict[str, MetricManager] = defaultdict(MetricManager)

    def _get_manager(self, metric_name: str) -> MetricManager:
        return self._managers[metric_name]

    @property
    def metric_names(self) -> [str]:
        return self._managers.keys()

    def add_batch_value(
        self,
        metric_name: str,
        value: float,
        batch_size: int
    ):
        self._get_manager(metric_name).add_batch_value(
            value=value,
            batch_size=batch_size
        )

    def begin_epoch(self):
        for manager in self._managers.values():
            manager.begin_epoch()

    def end_epoch(self):
        for manager in self._managers.values():
            manager.end_epoch()

    def get_last_batch_value(
        self,
        metric_name: str
    ) -> float:
        return self._get_manager(metric_name).last_batch_value

    def get_all_last_batch_values(self) -> Dict[str, float]:
        return {k: self.get_last_batch_value(k) for k in self.metric_names}

    # def get_current_epoch_value(
    #     self,
    #     metric_name: str
    # ) -> float:
    #     return self._get_manager(metric_name).current_epoch_value

    # @property
    # def current_epoch_metrics_values(self) -> Dict[str, float]:
    #     return {k: self.get_current_epoch_value(k) for k in self.metric_names}

    def get_last_epoch_value(
        self,
        metric_name: str
    ) -> float:
        return self._get_manager(metric_name).last_epoch_value

    def get_all_metrics_last_epoch_values(self) -> Dict[str, float]:
        return {k: self.get_last_epoch_value(k) for k in self.metric_names}

    def get_all_epochs_values(self, metric_name: str) -> [float]:
        return self._get_manager(metric_name).all_epoch_values

    def get_best_epoch_value(
        self,
        metric_name: str,
        minimize: bool
    ) -> float:
        return self._get_manager(metric_name).get_best_epoch_value(minimize=minimize)


class Meter:

    def __init__(self):
        self._managers: Dict[str, MetricsManager] = defaultdict(MetricsManager)

    def _get_manager(self, phase: str) -> MetricsManager:
        return self._managers[phase]

    # def get_all_metric_names(self, phase: str) -> [str]:
    #     return self._managers[phase].metric_names

    def add_batch_value(
        self,
        phase: str,
        metric_name: str,
        value: float,
        batch_size: int
    ):
        self._get_manager(phase).add_batch_value(
            metric_name=metric_name,
            value=value,
            batch_size=batch_size
        )

    def begin_phase(self, phase: str):
        self._get_manager(phase).begin_epoch()

    def end_phase(self, phase: str):
        self._get_manager(phase).end_epoch()

    def get_last_batch_value(
        self,
        phase: str,
        metric_name: str
    ) -> float:
        return self._get_manager(phase).get_last_batch_value(metric_name)

    def get_all_last_batch_values(self, phase: str) -> Dict[str, float]:
        return self._get_manager(phase).get_all_last_batch_values()

    # Epochs

    def get_current_epoch_value(
        self,
        phase: str,
        metric_name: str
    ) -> float:
        return self._get_manager(phase).get_current_epoch_value(metric_name)

    # def get_current_epoch_metrics_values(
    #     self,
    #     phase: str
    # ) -> Dict[str, float]:
    #     return self._get_manager(phase).current_epoch_metrics_values

    def get_last_epoch_metrics_values(
        self,
        phase: str
    ) -> Dict[str, float]:
        return self._get_manager(phase).get_all_metrics_last_epoch_values()

    def get_last_epoch_value(
        self,
        phase: str,
        metric_name: str
    ) -> float:
        return self._get_manager(phase).get_last_epoch_value(metric_name)

    def get_all_epochs_values(
        self,
        phase: str,
        metric_name: str
    ) -> [float]:
        return self._get_manager(phase).get_all_epochs_values(metric_name)

    def get_best_epoch_value(
        self,
        phase: str,
        metric_name: str,
        minimize: bool
    ) -> float:
        return self._get_manager(phase).get_best_epoch_value(
            metric_name=metric_name,
            minimize=minimize
        )

    def is_last_epoch_value_best(
        self,
        phase: str,
        metric_name: str,
        minimize: bool
    ) -> bool:
        last = self.get_last_epoch_value(
            phase=phase,
            metric_name=metric_name
        )

        best = self.get_best_epoch_value(
            phase=phase,
            metric_name=metric_name,
            minimize=minimize
        )

        return last == best




class Monitor:

    def __init__(self, str: str):
        components = str.split('_')

        self.str = str
        self.phase = components[0]
        self.metric_name = components[1]


class AverageMeter(object):

    def __init__(self):
        self.sum: float = 0.0
        self.count: int = 0

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def add(self, value: float, n: int):
        self.sum += (value * n)
        self.count += n

    def value(self):
        return self.sum / self.count
