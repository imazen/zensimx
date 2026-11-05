// Feature-gated observer: the plan and arithmetic belong to the bake scorer.

/// A revision-7 detector sampled on its native scale grid, before pooling.
/// Empty values identify a global statistic with no spatial detector.
#[derive(Clone, Debug)]
pub struct DetectorMap {
    /// Registry slot, preserving input identity.
    pub id: usize,
    /// Display grid width (long edge capped at 512).
    pub width: usize,
    /// Display grid height.
    pub height: usize,
    /// Full padded pooling grid; display samples do not re-pool the feature.
    pub pooling_grid: [usize; 2],
    /// Full detector's extrema, including the pooling boundary.
    pub range: [f32; 2],
    /// Detector or unsupported/global explanation.
    pub role: String,
    /// Row-major f32 values without quantization.
    pub values: Vec<f32>,
    #[cfg_attr(not(feature = "viz-export"), allow(dead_code))]
    pub(crate) file_name: String,
}

/// In-memory retained planes for one pair. A map request evaluates its own
/// scale/channel; it does not retain a second collection of all detector maps.
pub struct DetectorAtlas {
    retention: crate::feature_v2::FoldRetention,
    original: (usize, usize),
    slots: SlotSet,
}
impl DetectorAtlas {
    /// Native formula operands at one displayed basic-detector cell. The shared
    /// file observer checks the same retained SSIM value bit for bit.
    pub fn basic_terms(&self, id: usize, x: usize, y: usize) -> Result<[f32; 13], ResearchError> {
        if id >= 156 || !self.slots.contains(id) {
            return Err(ResearchError::Plan("basic-family detector required".into()));
        }
        let def = crate::feature_defs::def_at(id, crate::NUM_SCALES)
            .ok_or_else(|| ResearchError::Plan("unknown slot".into()))?;
        let scale = usize::from(def.scale);
        let ch = ["x", "y", "b"]
            .iter()
            .position(|&c| c == def.channel.as_str())
            .ok_or_else(|| ResearchError::Plan("unknown channel".into()))?;
        let (w, h) = self.retention.dims[scale];
        let logical = [
            (self.original.0 >> scale).max(1).min(w),
            (self.original.1 >> scale).max(1).min(h),
        ];
        let step = logical[0].max(logical[1]).div_ceil(512);
        if x >= logical[0].div_ceil(step) || y >= logical[1].div_ceil(step) {
            return Err(ResearchError::Plan("cell outside map".into()));
        }
        crate::feature_v2::viz_basic_cell(&self.retention, scale, ch, y * step * w + x * step)
            .map_err(|e| ResearchError::Plan(e.to_string()))
    }
    /// Evaluate a requested, populated detector using the file observer's owner.
    pub fn map(&self, id: usize) -> Result<DetectorMap, ResearchError> {
        if !self.slots.contains(id) {
            return Err(ResearchError::Plan(
                "detector slot is outside this bake".into(),
            ));
        }
        let mut result = None;
        crate::feature_v2::viz_visit_maps(
            &self.retention,
            &SlotSet::from_slots([id]),
            self.original,
            |map| {
                result = Some(map);
                Ok(())
            },
        )
        .map_err(|e| ResearchError::Plan(e.to_string()))?;
        result.ok_or_else(|| ResearchError::Plan("detector has no scale/channel".into()))
    }
}

/// Extract a bake's exact validated pixel plan, retaining planes in memory.
/// The existing extract path and its optional file observer remain unchanged.
/// Only revision 7 is supported. Returned features use registry ID positions,
/// as required by BakeScorer::score_features, including for dense bake layouts.
pub fn extract_with_detectors(
    bake: &[u8],
    source: &impl ImageSource,
    distorted: &impl ImageSource,
) -> Result<(Vec<f64>, DetectorAtlas), ResearchError> {
    let model = crate::mlp::Model::from_bytes(bake).map_err(|_| ResearchError::UnreadableBake)?;
    crate::BakeScorer::new(&model).map_err(|e| ResearchError::Plan(e.to_string()))?;
    let plan = Plan::for_bake(&model)?;
    if plan.compute.formula_revision != crate::feature_defs::FormulaRevision::Rev7 {
        return Err(ResearchError::Plan(
            "detector atlas requires revision 7".into(),
        ));
    }
    let req = Request::for_bake_bytes(bake)?;
    // BakeScorer::score_features accepts raw identity rows and performs the
    // bake's dense gather itself. Retain the validated revision, but emit the
    // research request's registry-ID layout rather than packed caller positions.
    let plan =
        Plan::derive_with_layout_at(req.want(), req.layout(), plan.compute.formula_revision)?;
    let slots = plan.emit.clone();
    let mut retention = None;
    let extraction = extract_planned(
        &req,
        source,
        distorted,
        None,
        Some(plan),
        Some(&mut retention),
    )?;
    Ok((
        extraction.into_values(),
        DetectorAtlas {
            retention: retention
                .ok_or_else(|| ResearchError::Plan("missing retained planes".into()))?,
            original: (source.width(), source.height()),
            slots,
        },
    ))
}
