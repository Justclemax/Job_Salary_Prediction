//! Rust inference server for the salary models.
//!
//! The Python side (`uv run python -m salary.export`) writes a JSON bundle in
//! `models/rust/`: the LightGBM point model (log target) and the two quantile models of
//! the conformalized 80 % interval. This crate loads it once at start-up and serves
//! predictions over HTTP with no Python, no native ML library and a few MB of memory.

pub mod forest;
pub mod schema;

use std::{path::Path, sync::Arc};

use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::Serialize;
use serde_json::{Value, json};

use crate::{forest::Forest, schema::Schema};

/// Largest batch accepted by `POST /predict`.
pub const MAX_BATCH: usize = 10_000;
/// Request body limit: a full batch is about 2.5 MB of JSON.
pub const MAX_BODY_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug)]
pub struct Engine {
    pub schema: Schema,
    pub point: Forest,
    pub low: Forest,
    pub high: Forest,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Prediction {
    pub salary: f64,
    pub low: f64,
    pub high: f64,
}

impl Engine {
    pub fn load(dir: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(dir.join("schema.json"))
            .map_err(|e| format!("{} : {e}", dir.join("schema.json").display()))?;
        let schema: Schema =
            serde_json::from_str(&text).map_err(|e| format!("schema.json : {e}"))?;
        let engine = Self {
            point: Forest::load(&dir.join("lightgbm.json"))?,
            low: Forest::load(&dir.join("quantile_low.json"))?,
            high: Forest::load(&dir.join("quantile_high.json"))?,
            schema,
        };
        let n = engine.schema.features.len();
        if [&engine.point, &engine.low, &engine.high]
            .iter()
            .any(|f| f.n_features() != n)
        {
            return Err("le schéma et les forêts n'ont pas le même nombre de variables".into());
        }
        Ok(engine)
    }

    /// Point estimate and 80 % interval, with the same arithmetic as the Streamlit simulator.
    pub fn predict(&self, x: &[f64]) -> Prediction {
        let offset = self.schema.interval.conformal_offset;
        let salary = self.point.predict(x);
        let low = (self.low.predict(x) - offset).max(0.0).min(salary);
        let high = (self.high.predict(x) + offset).max(salary);
        Prediction { salary, low, high }
    }

    /// Batch prediction: tree-major evaluation, split across CPU cores for large batches.
    /// Bit-identical to calling `predict` on each row.
    pub fn predict_batch(&self, rows: &[Vec<f64>]) -> Vec<Prediction> {
        let nf = self.schema.features.len();
        let flat: Vec<f64> = rows.iter().flatten().copied().collect();
        assert_eq!(flat.len(), rows.len() * nf, "nombre de variables");
        let mut out = vec![
            Prediction {
                salary: 0.0,
                low: 0.0,
                high: 0.0
            };
            rows.len()
        ];
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
        let chunk = rows.len().div_ceil(threads).max(256);
        std::thread::scope(|scope| {
            for (x, o) in flat.chunks(chunk * nf).zip(out.chunks_mut(chunk)) {
                scope.spawn(move || self.predict_chunk(x, o));
            }
        });
        out
    }

    fn predict_chunk(&self, rows: &[f64], out: &mut [Prediction]) {
        let n = out.len();
        let (mut p, mut l, mut h) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        self.point.add_raw_batch(rows, &mut p);
        self.low.add_raw_batch(rows, &mut l);
        self.high.add_raw_batch(rows, &mut h);
        let offset = self.schema.interval.conformal_offset;
        for (i, o) in out.iter_mut().enumerate() {
            let salary = self.point.finish(p[i]);
            let low = (self.low.finish(l[i]) - offset).max(0.0).min(salary);
            let high = (self.high.finish(h[i]) + offset).max(salary);
            *o = Prediction { salary, low, high };
        }
    }

    pub fn predict_value(&self, profile: &Value) -> Result<Prediction, String> {
        let obj = profile
            .as_object()
            .ok_or("chaque profil doit être un objet JSON")?;
        Ok(self.predict(&self.schema.encode(obj)?))
    }

    pub fn info(&self) -> Value {
        let forest = |f: &Forest| {
            json!({"name": f.name, "objective": f.objective, "trees": f.n_trees(),
                   "nodes": f.n_nodes(), "heap_bytes": f.heap_bytes()})
        };
        json!({
            "model": self.schema.model,
            "interval": self.schema.interval,
            "forests": [forest(&self.point), forest(&self.low), forest(&self.high)],
            "features": self.schema.features,
            "max_batch": MAX_BATCH,
        })
    }
}

fn error(status: StatusCode, msg: String, index: Option<usize>) -> Response {
    (status, Json(json!({"error": msg, "index": index}))).into_response()
}

async fn health() -> Json<Value> {
    Json(json!({"status": "ok"}))
}

async fn model(State(engine): State<Arc<Engine>>) -> Json<Value> {
    Json(engine.info())
}

/// Body: one profile object -> one prediction; an array of profiles -> `{"predictions": [...]}`.
async fn predict(State(engine): State<Arc<Engine>>, Json(body): Json<Value>) -> Response {
    match body {
        Value::Array(items) => {
            if items.len() > MAX_BATCH {
                let msg = format!("au plus {MAX_BATCH} profils par requête");
                return error(StatusCode::PAYLOAD_TOO_LARGE, msg, None);
            }
            let mut rows = Vec::with_capacity(items.len());
            for (i, item) in items.iter().enumerate() {
                let encoded = item
                    .as_object()
                    .ok_or_else(|| "chaque profil doit être un objet JSON".to_string())
                    .and_then(|o| engine.schema.encode(o));
                match encoded {
                    Ok(x) => rows.push(x),
                    Err(e) => return error(StatusCode::UNPROCESSABLE_ENTITY, e, Some(i)),
                }
            }
            // CPU-bound work runs off the async runtime threads.
            let predictions = if rows.len() < 64 {
                rows.iter().map(|x| engine.predict(x)).collect()
            } else {
                let engine = engine.clone();
                match tokio::task::spawn_blocking(move || engine.predict_batch(&rows)).await {
                    Ok(p) => p,
                    Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string(), None),
                }
            };
            Json(json!({"predictions": predictions})).into_response()
        }
        other => match engine.predict_value(&other) {
            Ok(p) => Json(p).into_response(),
            Err(e) => error(StatusCode::UNPROCESSABLE_ENTITY, e, None),
        },
    }
}

pub fn router(engine: Arc<Engine>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/model", get(model))
        .route("/predict", post(predict))
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .with_state(engine)
}
