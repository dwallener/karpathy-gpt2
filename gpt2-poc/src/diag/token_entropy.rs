#[derive(Debug, Clone)]
pub struct TokenEntropyReport {
    pub entropy: f32,
    pub top1_prob: f32,
    pub top10_mass: f32,
    pub probs: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct AggregateTokenEntropyReport {
    pub entropy: f32,
    pub top1_prob: f32,
    pub top10_mass: f32,
    pub probs: Vec<f32>,
    pub samples: usize,
}

pub fn analyze_logits(logits: &[f32], top_k: usize) -> TokenEntropyReport {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp = logits.iter().map(|x| (x - max).exp()).collect::<Vec<_>>();
    let sum = exp.iter().sum::<f32>().max(1e-12);
    let probs = exp.iter().map(|x| x / sum).collect::<Vec<_>>();

    let mut probs_sorted = probs.clone();
    probs_sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let keep = top_k.min(probs_sorted.len());
    let top_probs = probs_sorted[..keep].to_vec();

    let entropy = probs
        .iter()
        .filter(|p| **p > 0.0)
        .map(|p| -p * p.ln())
        .sum::<f32>();
    let top1_prob = probs_sorted.first().copied().unwrap_or(0.0);
    let top10_mass = probs_sorted.iter().take(10).sum::<f32>();

    TokenEntropyReport {
        entropy,
        top1_prob,
        top10_mass,
        probs: top_probs,
    }
}

pub fn aggregate_reports(reports: &[TokenEntropyReport], top_k: usize) -> Option<AggregateTokenEntropyReport> {
    if reports.is_empty() {
        return None;
    }
    let keep = top_k.min(reports.iter().map(|report| report.probs.len()).min().unwrap_or(0));
    let mut probs = vec![0.0; keep];
    let mut entropy = 0.0;
    let mut top1_prob = 0.0;
    let mut top10_mass = 0.0;

    for report in reports {
        entropy += report.entropy;
        top1_prob += report.top1_prob;
        top10_mass += report.top10_mass;
        for (idx, prob) in report.probs.iter().take(keep).enumerate() {
            probs[idx] += *prob;
        }
    }

    let count = reports.len() as f32;
    for prob in &mut probs {
        *prob /= count;
    }

    Some(AggregateTokenEntropyReport {
        entropy: entropy / count,
        top1_prob: top1_prob / count,
        top10_mass: top10_mass / count,
        probs,
        samples: reports.len(),
    })
}

pub fn ascii_rank_plot(probs: &[f32]) -> String {
    let mut lines = Vec::new();
    lines.push("Token Prob vs Rank".to_string());
    lines.push(String::new());
    let top1 = probs.first().copied().unwrap_or(1e-6).max(1e-6);
    for (idx, prob) in probs.iter().enumerate() {
        let stars = ((prob / top1) * 40.0).round().clamp(1.0, 40.0) as usize;
        lines.push(format!("{:>2} | {}", idx + 1, "*".repeat(stars)));
    }
    lines.join("\n")
}
