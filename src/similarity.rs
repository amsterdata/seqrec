use serde::{Serialize, Deserialize};

pub trait Similarity {
    fn from_norms(&self, dot_product: f64, norm_a: f64, norm_b: f64) -> f64;
    fn accumulate_norm(&self, value: f64) -> f64;
    fn finalize_norm(&self, accum: f64) -> f64;
    fn update_norm(&self, old_norm: f64, value_to_delete: f64) -> f64;
}

pub const COSINE: Cosine = Cosine {};

#[derive(Serialize, Deserialize)]
pub struct Cosine {}

impl Similarity for Cosine {

    #[inline(always)]
    fn from_norms(&self, dot_product: f64, norm_a: f64, norm_b: f64) -> f64 {
        dot_product / (norm_a * norm_b)
    }

    fn accumulate_norm(&self, value: f64) -> f64 {
        value * value
    }

    fn finalize_norm(&self, accum: f64) -> f64 {
        accum.sqrt()
    }

    fn update_norm(&self, old_norm: f64, value_to_delete: f64) -> f64 {
        ((old_norm * old_norm) - (value_to_delete * value_to_delete)).sqrt()
    }
}