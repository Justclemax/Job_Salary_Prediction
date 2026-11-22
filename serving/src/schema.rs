//! Input validation and encoding: raw profile (JSON object) -> feature vector.
//!
//! Mirrors the Python `tree` preprocessor: numerics passed through, categories replaced
//! by their index in the level list of `salary.config` (exported in `schema.json`).

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Feature {
    Numeric { name: String, min: f64, max: f64 },
    Categorical { name: String, levels: Vec<String> },
}

impl Feature {
    pub fn name(&self) -> &str {
        match self {
            Feature::Numeric { name, .. } | Feature::Categorical { name, .. } => name,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Interval {
    pub alpha_low: f64,
    pub alpha_high: f64,
    pub conformal_offset: f64,
    pub coverage: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Schema {
    pub format: String,
    pub features: Vec<Feature>,
    pub interval: Interval,
    pub model: Value,
}

impl Schema {
    /// Encode one profile. Every feature is required; unknown keys are rejected.
    pub fn encode(&self, profile: &Map<String, Value>) -> Result<Vec<f64>, String> {
        if let Some(extra) = profile
            .keys()
            .find(|k| !self.features.iter().any(|f| f.name() == k.as_str()))
        {
            return Err(format!("variable inconnue : {extra}"));
        }
        self.features
            .iter()
            .map(|f| {
                let v = profile
                    .get(f.name())
                    .ok_or_else(|| format!("variable manquante : {}", f.name()))?;
                match f {
                    Feature::Numeric { name, min, max } => {
                        let x = v
                            .as_f64()
                            .ok_or_else(|| format!("{name} doit être un nombre"))?;
                        if !(x >= *min && x <= *max) {
                            return Err(format!(
                                "{name} = {x} hors de l'intervalle [{min}, {max}]"
                            ));
                        }
                        Ok(x)
                    }
                    Feature::Categorical { name, levels } => {
                        let s = v
                            .as_str()
                            .ok_or_else(|| format!("{name} doit être une chaîne"))?;
                        levels
                            .iter()
                            .position(|l| l == s)
                            .map(|i| i as f64)
                            .ok_or_else(|| format!("modalité inconnue pour {name} : {s}"))
                    }
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn schema() -> Schema {
        serde_json::from_value(json!({
            "format": "salary-forest/1",
            "features": [
                {"kind": "numeric", "name": "experience_years", "min": 0, "max": 20},
                {"kind": "categorical", "name": "education_level", "levels": ["Bachelor", "Master", "PhD"]}
            ],
            "interval": {"alpha_low": 0.1, "alpha_high": 0.9, "conformal_offset": 10.0, "coverage": 80.0},
            "model": {"name": "lightgbm"}
        }))
        .unwrap()
    }

    fn obj(v: Value) -> Map<String, Value> {
        v.as_object().unwrap().clone()
    }

    #[test]
    fn encodes_in_schema_order() {
        let x = schema()
            .encode(&obj(
                json!({"education_level": "PhD", "experience_years": 7}),
            ))
            .unwrap();
        assert_eq!(x, vec![7.0, 2.0]);
    }

    #[test]
    fn rejects_invalid_profiles() {
        let s = schema();
        let err = |v| s.encode(&obj(v)).unwrap_err();
        assert!(err(json!({"experience_years": 7})).contains("manquante"));
        assert!(err(json!({"experience_years": 25, "education_level": "PhD"})).contains("hors"));
        assert!(err(json!({"experience_years": 7, "education_level": "MBA"})).contains("inconnue"));
        assert!(err(json!({"experience_years": "7", "education_level": "PhD"})).contains("nombre"));
        assert!(
            err(json!({"experience_years": 7, "education_level": "PhD", "age": 30}))
                .contains("inconnue")
        );
    }
}
