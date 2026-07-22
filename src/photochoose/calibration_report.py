from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

from photochoose.analysis_profiles import get_analysis_profile
from photochoose.database import PhotoDatabase

MIN_SUGGESTION_SAMPLES = 5
LARGE_CORPUS_DECISION_TARGET = 30
LARGE_CORPUS_BEST_TARGET = 20

SIGNAL_LABELS = {
    "focus": "global focus",
    "exposure": "exposure",
    "faces": "face focus/detection",
    "eyes": "eye state/confidence",
    "objects": "primary-object focus",
    "grouping": "duplicate/burst separation",
}


@dataclass(frozen=True)
class ProfileCalibrationSummary:
    profile: str
    version: str
    actions: int
    decision_disagreements: int
    rescued_to_keep: int
    downgraded_from_keep: int
    best_corrections: int


@dataclass(frozen=True)
class SignalCalibrationSummary:
    signal: str
    disagreements: int
    rescued_to_keep: int
    downgraded_from_keep: int


@dataclass(frozen=True)
class CalibrationSuggestion:
    profile: str
    signal: str
    sample_count: int
    message: str


@dataclass(frozen=True)
class CalibrationReport:
    project_name: str
    total_actions: int
    decision_actions: int
    decision_disagreements: int
    best_actions: int
    best_corrections: int
    directions: Tuple[Tuple[str, int], ...]
    profiles: Tuple[ProfileCalibrationSummary, ...]
    signals: Tuple[SignalCalibrationSummary, ...]
    suggestions: Tuple[CalibrationSuggestion, ...]
    recent_examples: Tuple[str, ...]

    @property
    def has_enough_decision_data(self) -> bool:
        return self.decision_disagreements >= LARGE_CORPUS_DECISION_TARGET

    @property
    def has_enough_best_data(self) -> bool:
        return self.best_corrections >= LARGE_CORPUS_BEST_TARGET

    def to_text(self) -> str:
        lines = [
            "Private calibration report · {0}".format(self.project_name),
            "",
            "This report uses local manual-action snapshots only. It does not inspect "
            "or upload photographs, and it never changes profile thresholds automatically.",
            "",
            "Coverage",
            "  Recorded manual actions: {0}".format(self.total_actions),
            "  Decision actions: {0}".format(self.decision_actions),
            "  Automatic decision disagreements: {0}".format(
                self.decision_disagreements
            ),
            "  Best actions: {0}".format(self.best_actions),
            "  Automatic Best corrections: {0}".format(self.best_corrections),
            "",
        ]
        if self.has_enough_decision_data and self.has_enough_best_data:
            lines.append(
                "Corpus status: enough recorded corrections for an initial threshold review; "
                "confirm on the target machine before changing defaults."
            )
        else:
            lines.append(
                "Corpus status: early data. Large real-shoot validation is still required on "
                "the target machine before tuning defaults."
            )

        if self.directions:
            lines.extend(("", "Decision disagreements"))
            for direction, count in self.directions:
                lines.append("  {0}: {1}".format(direction, count))

        if self.profiles:
            lines.extend(("", "By analysis profile"))
            for summary in self.profiles:
                lines.append(
                    "  {0} / {1}: {2} actions · {3} decision disagreements · "
                    "{4} rescued to Keep · {5} downgraded from Keep · "
                    "{6} Best corrections".format(
                        _profile_label(summary.profile),
                        summary.version,
                        summary.actions,
                        summary.decision_disagreements,
                        summary.rescued_to_keep,
                        summary.downgraded_from_keep,
                        summary.best_corrections,
                    )
                )

        if self.signals:
            lines.extend(("", "Signals present in decision disagreements"))
            for summary in self.signals:
                lines.append(
                    "  {0}: {1} · {2} rescued to Keep · {3} downgraded from Keep".format(
                        SIGNAL_LABELS[summary.signal],
                        summary.disagreements,
                        summary.rescued_to_keep,
                        summary.downgraded_from_keep,
                    )
                )

        lines.extend(("", "Provisional suggestions"))
        if self.suggestions:
            for suggestion in self.suggestions:
                lines.append("  • " + suggestion.message)
        else:
            lines.append(
                "  No threshold suggestion has reached the minimum of {0} matching "
                "disagreements.".format(MIN_SUGGESTION_SAMPLES)
            )

        if self.recent_examples:
            lines.extend(("", "Recent correction examples"))
            lines.extend("  " + example for example in self.recent_examples)

        lines.extend(
            (
                "",
                "Next validation step: run representative large shoots on the target machine, "
                "make normal final choices, then compare this report across profiles.",
            )
        )
        return "\n".join(lines)


def build_calibration_report(
    database: PhotoDatabase, project_id: int
) -> CalibrationReport:
    project = database.get_project(project_id)
    if project is None:
        raise ValueError("Unknown project id: {0}".format(project_id))
    actions = database.list_manual_overrides(project_id)
    decision_actions = [item for item in actions if item["action_type"] == "decision"]
    best_actions = [item for item in actions if item["action_type"] == "best"]
    disagreements = [
        item
        for item in decision_actions
        if item["previous_source"] == "automatic"
        and item["previous_value"] != item["new_value"]
    ]
    best_corrections = [
        item for item in best_actions if item["previous_source"] == "automatic"
    ]

    directions = Counter(
        "{0} → {1}".format(
            str(item["previous_value"]).title(),
            str(item["new_value"]).title(),
        )
        for item in disagreements
    )
    profile_summaries = _profile_summaries(actions, disagreements, best_corrections)
    signal_summaries = _signal_summaries(disagreements)
    suggestions = _suggestions(disagreements, best_corrections)
    relevant_examples = disagreements + best_corrections
    relevant_examples.sort(key=lambda item: int(item["id"]), reverse=True)
    recent_examples = tuple(
        _example_line(item) for item in relevant_examples[:10]
    )
    return CalibrationReport(
        project_name=project.name,
        total_actions=len(actions),
        decision_actions=len(decision_actions),
        decision_disagreements=len(disagreements),
        best_actions=len(best_actions),
        best_corrections=len(best_corrections),
        directions=tuple(sorted(directions.items(), key=lambda item: (-item[1], item[0]))),
        profiles=profile_summaries,
        signals=signal_summaries,
        suggestions=suggestions,
        recent_examples=recent_examples,
    )


def _profile_summaries(
    actions: Sequence[Mapping[str, object]],
    disagreements: Sequence[Mapping[str, object]],
    best_corrections: Sequence[Mapping[str, object]],
) -> Tuple[ProfileCalibrationSummary, ...]:
    grouped_actions = Counter(_profile_key(item) for item in actions)
    grouped_disagreements = Counter(_profile_key(item) for item in disagreements)
    grouped_rescues = Counter(
        _profile_key(item) for item in disagreements if item["new_value"] == "keep"
    )
    grouped_downgrades = Counter(
        _profile_key(item)
        for item in disagreements
        if item["previous_value"] == "keep" and item["new_value"] != "keep"
    )
    grouped_best = Counter(_profile_key(item) for item in best_corrections)
    summaries = []
    for profile, version in sorted(grouped_actions):
        key = (profile, version)
        summaries.append(
            ProfileCalibrationSummary(
                profile=profile,
                version=version,
                actions=grouped_actions[key],
                decision_disagreements=grouped_disagreements[key],
                rescued_to_keep=grouped_rescues[key],
                downgraded_from_keep=grouped_downgrades[key],
                best_corrections=grouped_best[key],
            )
        )
    return tuple(summaries)


def _signal_summaries(
    disagreements: Sequence[Mapping[str, object]],
) -> Tuple[SignalCalibrationSummary, ...]:
    totals = Counter()
    rescues = Counter()
    downgrades = Counter()
    for item in disagreements:
        for signal in _signals_for(item):
            totals[signal] += 1
            if item["new_value"] == "keep":
                rescues[signal] += 1
            if item["previous_value"] == "keep" and item["new_value"] != "keep":
                downgrades[signal] += 1
    return tuple(
        SignalCalibrationSummary(
            signal=signal,
            disagreements=totals[signal],
            rescued_to_keep=rescues[signal],
            downgraded_from_keep=downgrades[signal],
        )
        for signal in SIGNAL_LABELS
        if totals[signal]
    )


def _suggestions(
    disagreements: Sequence[Mapping[str, object]],
    best_corrections: Sequence[Mapping[str, object]],
) -> Tuple[CalibrationSuggestion, ...]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, object]]] = defaultdict(list)
    for item in disagreements:
        profile, _version = _profile_key(item)
        for signal in _signals_for(item):
            grouped[(profile, signal)].append(item)

    suggestions = []
    for (profile, signal), items in sorted(grouped.items()):
        if len(items) < MIN_SUGGESTION_SAMPLES:
            continue
        rescued = sum(item["new_value"] == "keep" for item in items)
        downgraded = sum(
            item["previous_value"] == "keep" and item["new_value"] != "keep"
            for item in items
        )
        if rescued / float(len(items)) >= 0.6:
            message = (
                "{0}: {1} rules may be too cautious ({2} of {3} matching automatic "
                "decisions were changed to Keep). Review on the target corpus."
            ).format(
                _profile_label(profile),
                SIGNAL_LABELS[signal],
                rescued,
                len(items),
            )
        elif downgraded / float(len(items)) >= 0.6:
            message = (
                "{0}: {1} rules may be too permissive ({2} of {3} matching automatic "
                "Keep decisions were downgraded). Review on the target corpus."
            ).format(
                _profile_label(profile),
                SIGNAL_LABELS[signal],
                downgraded,
                len(items),
            )
        else:
            continue
        suggestions.append(
            CalibrationSuggestion(
                profile=profile,
                signal=signal,
                sample_count=len(items),
                message=message,
            )
        )

    if len(best_corrections) >= MIN_SUGGESTION_SAMPLES:
        suggestions.append(
            CalibrationSuggestion(
                profile="all",
                signal="grouping",
                sample_count=len(best_corrections),
                message=(
                    "Best ranking has {0} corrections. Inspect eye/focus weighting and "
                    "diversity on the target corpus before changing it."
                ).format(len(best_corrections)),
            )
        )
    return tuple(suggestions)


def _signals_for(item: Mapping[str, object]) -> Tuple[str, ...]:
    reasons = " ".join(str(value).casefold() for value in item["auto_reasons"])
    signals = []
    if any(word in reasons for word in ("detail", "blur", "soft", "focus")):
        signals.append("focus")
    if any(word in reasons for word in ("dark", "bright", "exposure", "clipped")):
        signals.append("exposure")
    if "face" in reasons:
        signals.append("faces")
    if "eye" in reasons:
        signals.append("eyes")
    if any(word in reasons for word in ("primary detected subject", "object")):
        signals.append("objects")
    if any(word in reasons for word in ("duplicate", "weaker", "burst", "group")):
        signals.append("grouping")
    return tuple(signals)


def _profile_key(item: Mapping[str, object]) -> Tuple[str, str]:
    return (
        str(item["analysis_profile"] or "unknown"),
        str(item["analysis_version"] or "legacy"),
    )


def _profile_label(profile: str) -> str:
    try:
        return get_analysis_profile(profile).label
    except ValueError:
        return profile.title()


def _example_line(item: Mapping[str, object]) -> str:
    path = str(item["photo_path"] or "Photo #{0}".format(item["photo_id"]))
    if item["action_type"] == "best":
        change = "Best {0} → {1}".format(
            str(item["previous_value"]), str(item["new_value"])
        )
    else:
        change = "{0} → {1}".format(
            str(item["previous_value"]).title(),
            str(item["new_value"]).title(),
        )
    profile, version = _profile_key(item)
    return "{0} · {1} · {2} / {3}".format(
        path, change, _profile_label(profile), version
    )
