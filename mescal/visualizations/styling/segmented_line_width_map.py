import numpy as np


class LineWidthMap:
    """Maps values to line widths using customizable segments similar to SegmentedColorMap."""

    def __init__(
            self,
            segments: dict[tuple[float, float], float | list[float]],
            na_width: float = 1.0
    ):
        self.segments_dict = segments
        self.sorted_segments = self._sort_segments(segments)
        self.min_value = self.sorted_segments[0][0][0]
        self.max_value = self.sorted_segments[-1][0][1]
        self.na_width = na_width

    def _sort_segments(self, segments):
        sorted_segments = sorted(segments.items(), key=lambda x: x[0][0])
        prev_end = -float('inf')
        for (start, end), _ in sorted_segments:
            if start < prev_end:
                raise ValueError(
                    f"Overlapping segments detected: ({start}, {end}) overlaps with previous segment ending at {prev_end}")
            prev_end = end
        return sorted_segments

    def __call__(self, value: float) -> float:
        """Get the appropriate width for a value."""
        if np.isnan(value):
            return self.na_width

        for (start, end), width_data in self.sorted_segments:
            if start <= value <= end:
                if isinstance(width_data, (int, float)):
                    return width_data

                # Handle list of widths with linear interpolation
                width_list = width_data if isinstance(width_data, list) else [width_data]
                if len(width_list) == 1:
                    return width_list[0]

                # Interpolate between width points
                pos_in_segment = (value - start) / (end - start)
                pos_idx = pos_in_segment * (len(width_list) - 1)
                idx_low = int(np.floor(pos_idx))
                idx_high = int(np.ceil(pos_idx))

                if idx_low == idx_high:
                    return width_list[idx_low]

                # Linear interpolation
                width_low = width_list[idx_low]
                width_high = width_list[idx_high]
                frac = pos_idx - idx_low
                return width_low + frac * (width_high - width_low)

        # Default for values outside ranges
        if value < self.min_value:
            first_width = self.sorted_segments[0][1]
            return first_width[0] if isinstance(first_width, list) else first_width
        last_width = self.sorted_segments[-1][1]
        return last_width[-1] if isinstance(last_width, list) else last_width