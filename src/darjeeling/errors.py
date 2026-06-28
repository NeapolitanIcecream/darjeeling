from __future__ import annotations

from typing import Any


class DarjeelingError(Exception):
    """Base class for Darjeeling errors."""


class ValidationError(DarjeelingError):
    pass


class TargetDefinitionError(DarjeelingError):
    pass


class SnapshotBuildError(DarjeelingError):
    def __init__(
        self,
        message: str,
        *,
        reference_qualification: Any | None = None,
        reference_failure_report: Any | None = None,
        reference_usage: Any | None = None,
    ) -> None:
        super().__init__(message)
        self.reference_qualification = reference_qualification
        self.reference_failure_report = reference_failure_report
        self.reference_usage = reference_usage


class ArtifactError(DarjeelingError):
    pass


class WorkspaceError(DarjeelingError):
    pass


class EvaluationError(DarjeelingError):
    pass


class ReleaseError(DarjeelingError):
    pass


class RuntimeErrorSafe(DarjeelingError):
    pass


class TelemetryError(DarjeelingError):
    pass


class CompileLaunchError(DarjeelingError):
    pass
