use crate::{EvalError, TimingStats, summarize_durations};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PipelineDurations<'a> {
    pub detect: &'a [std::time::Duration],
    pub track: &'a [std::time::Duration],
    pub recognize: &'a [std::time::Duration],
    pub end_to_end: &'a [std::time::Duration],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PipelineBenchmarkReport {
    pub frames: usize,
    pub detect: TimingStats,
    pub track: TimingStats,
    pub recognize: TimingStats,
    pub end_to_end: TimingStats,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct StageThresholds {
    pub min_fps: Option<f64>,
    pub max_mean_ms: Option<f64>,
    pub max_p95_ms: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PipelineBenchmarkThresholds {
    pub detect: StageThresholds,
    pub track: StageThresholds,
    pub recognize: StageThresholds,
    pub end_to_end: StageThresholds,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BenchmarkViolation {
    pub stage: &'static str,
    pub metric: &'static str,
    pub expected: f64,
    pub observed: f64,
}

pub fn summarize_pipeline_durations(
    durations: PipelineDurations<'_>,
) -> Result<PipelineBenchmarkReport, EvalError> {
    let frames = durations.end_to_end.len();
    if frames == 0 {
        return Err(EvalError::EmptyDurationSeries);
    }

    validate_series_len(frames, durations.detect.len(), "detect")?;
    validate_series_len(frames, durations.track.len(), "track")?;
    validate_series_len(frames, durations.recognize.len(), "recognize")?;

    Ok(PipelineBenchmarkReport {
        frames,
        detect: summarize_durations(durations.detect)?,
        track: summarize_durations(durations.track)?,
        recognize: summarize_durations(durations.recognize)?,
        end_to_end: summarize_durations(durations.end_to_end)?,
    })
}

pub fn parse_pipeline_benchmark_thresholds(
    text: &str,
) -> Result<PipelineBenchmarkThresholds, EvalError> {
    let mut thresholds = PipelineBenchmarkThresholds::default();

    for (line_idx, raw_line) in text.lines().enumerate() {
        let line_no = line_idx + 1;
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (key, value_str) =
            line.split_once('=')
                .ok_or_else(|| EvalError::InvalidThresholdEntry {
                    line: line_no,
                    message: "expected `stage.metric=value` entry".to_string(),
                })?;
        let (stage, metric) =
            key.split_once('.')
                .ok_or_else(|| EvalError::InvalidThresholdEntry {
                    line: line_no,
                    message: "expected key in form `stage.metric`".to_string(),
                })?;
        let value =
            value_str
                .trim()
                .parse::<f64>()
                .map_err(|_| EvalError::InvalidThresholdEntry {
                    line: line_no,
                    message: format!("invalid numeric value `{}`", value_str.trim()),
                })?;
        if !value.is_finite() || value < 0.0 {
            return Err(EvalError::InvalidThresholdEntry {
                line: line_no,
                message: format!(
                    "threshold value must be finite and non-negative, got {}",
                    value_str.trim()
                ),
            });
        }

        let stage_thresholds = match stage {
            "detect" => &mut thresholds.detect,
            "track" => &mut thresholds.track,
            "recognize" => &mut thresholds.recognize,
            "end_to_end" => &mut thresholds.end_to_end,
            _ => {
                return Err(EvalError::InvalidThresholdEntry {
                    line: line_no,
                    message: format!("unsupported stage `{stage}`"),
                });
            }
        };

        match metric {
            "min_fps" => stage_thresholds.min_fps = Some(value),
            "max_mean_ms" => stage_thresholds.max_mean_ms = Some(value),
            "max_p95_ms" => stage_thresholds.max_p95_ms = Some(value),
            _ => {
                return Err(EvalError::InvalidThresholdEntry {
                    line: line_no,
                    message: format!("unsupported metric `{metric}`"),
                });
            }
        }
    }

    Ok(thresholds)
}

pub fn validate_pipeline_benchmark_thresholds(
    report: &PipelineBenchmarkReport,
    thresholds: &PipelineBenchmarkThresholds,
) -> Vec<BenchmarkViolation> {
    let mut violations = Vec::new();
    validate_stage_thresholds(
        "detect",
        &report.detect,
        &thresholds.detect,
        &mut violations,
    );
    validate_stage_thresholds("track", &report.track, &thresholds.track, &mut violations);
    validate_stage_thresholds(
        "recognize",
        &report.recognize,
        &thresholds.recognize,
        &mut violations,
    );
    validate_stage_thresholds(
        "end_to_end",
        &report.end_to_end,
        &thresholds.end_to_end,
        &mut violations,
    );
    violations
}

const fn validate_series_len(
    expected: usize,
    got: usize,
    series: &'static str,
) -> Result<(), EvalError> {
    if expected != got {
        return Err(EvalError::DurationSeriesLengthMismatch {
            expected,
            got,
            series,
        });
    }
    Ok(())
}

fn validate_stage_thresholds(
    stage: &'static str,
    stats: &TimingStats,
    thresholds: &StageThresholds,
    violations: &mut Vec<BenchmarkViolation>,
) {
    if let Some(min_fps) = thresholds.min_fps
        && stats.throughput_fps < min_fps
    {
        violations.push(BenchmarkViolation {
            stage,
            metric: "min_fps",
            expected: min_fps,
            observed: stats.throughput_fps,
        });
    }
    if let Some(max_mean_ms) = thresholds.max_mean_ms
        && stats.mean_ms > max_mean_ms
    {
        violations.push(BenchmarkViolation {
            stage,
            metric: "max_mean_ms",
            expected: max_mean_ms,
            observed: stats.mean_ms,
        });
    }
    if let Some(max_p95_ms) = thresholds.max_p95_ms
        && stats.p95_ms > max_p95_ms
    {
        violations.push(BenchmarkViolation {
            stage,
            metric: "max_p95_ms",
            expected: max_p95_ms,
            observed: stats.p95_ms,
        });
    }
}
