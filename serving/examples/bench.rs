//! In-process benchmark: `cargo run --release --example bench` (needs models/rust/).

use std::{path::PathBuf, time::Instant};

use salary_serve::Engine;
use serde_json::Value;

fn main() {
    let dir = PathBuf::from(
        std::env::var("SALARY_MODEL_DIR").unwrap_or_else(|_| "../models/rust".into()),
    );
    let engine = Engine::load(&dir).expect("bundle");
    let parity: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("parity.json")).unwrap()).unwrap();
    let profiles: Vec<Value> = parity["profiles"]
        .as_array()
        .unwrap()
        .iter()
        .cycle()
        .take(10_000)
        .cloned()
        .collect();

    let t = Instant::now();
    let rows: Vec<Vec<f64>> = profiles
        .iter()
        .map(|p| engine.schema.encode(p.as_object().unwrap()).unwrap())
        .collect();
    let encode = t.elapsed();

    let t = Instant::now();
    let mut sum = 0.0;
    for x in &rows {
        sum += engine.predict(x).salary;
    }
    let predict = t.elapsed();

    let t = Instant::now();
    let batch = engine.predict_batch(&rows);
    let parallel = t.elapsed();
    assert!(
        batch
            .iter()
            .zip(&rows)
            .all(|(b, x)| *b == engine.predict(x))
    );

    let t = Instant::now();
    let body = serde_json::to_string(&profiles).unwrap();
    let _: Value = serde_json::from_str(&body).unwrap();
    let parse = t.elapsed();

    println!(
        "10 000 profils : parse JSON {parse:.1?}, encodage {encode:.1?}, prédiction ligne par ligne \
         {predict:.1?} ({:.1} µs/profil), lot multi-cœurs {parallel:.1?} (résultats identiques) \
         [checksum {sum:.0}]",
        predict.as_secs_f64() * 1e6 / 10_000.0
    );
}
