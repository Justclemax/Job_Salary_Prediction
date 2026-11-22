//! Rust must reproduce the Python predictions exactly (to 1e-9 relative).
//! Needs the bundle from `uv run python -m salary.export`; skipped otherwise.

use std::{path::PathBuf, sync::Arc};

use axum::{body::Body, http::Request};
use http_body_util::BodyExt;
use salary_serve::{Engine, router};
use serde_json::{Value, json};
use tower::ServiceExt;

fn bundle() -> Option<PathBuf> {
    let dir = std::env::var("SALARY_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models/rust"));
    if dir.join("parity.json").exists() {
        Some(dir)
    } else {
        eprintln!("bundle absent ({}) : test ignoré", dir.display());
        None
    }
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1.0)
}

#[test]
fn matches_python_predictions() {
    let Some(dir) = bundle() else { return };
    let engine = Engine::load(&dir).unwrap();
    let parity: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("parity.json")).unwrap()).unwrap();
    let profiles = parity["profiles"].as_array().unwrap();
    assert!(profiles.len() >= 100);
    let mut worst: f64 = 0.0;
    for (i, profile) in profiles.iter().enumerate() {
        let p = engine.predict_value(profile).unwrap();
        for (key, got) in [("salary", p.salary), ("low", p.low), ("high", p.high)] {
            let want = parity[key][i].as_f64().unwrap();
            worst = worst.max(rel(got, want));
            assert!(
                rel(got, want) < 1e-9,
                "profil {i}, {key} : Rust {got} vs Python {want}"
            );
        }
    }
    eprintln!("{} profils, écart relatif max {worst:.2e}", profiles.len());
}

async fn call(app: axum::Router, method: &str, uri: &str, body: Value) -> (u16, Value) {
    let req = Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let res = app.oneshot(req).await.unwrap();
    let status = res.status().as_u16();
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}

#[tokio::test]
async fn http_api() {
    let Some(dir) = bundle() else { return };
    let app = router(Arc::new(Engine::load(&dir).unwrap()));
    let profile = json!({
        "job_title": "Data Scientist", "experience_years": 5, "education_level": "Master",
        "skills_count": 8, "industry": "Technology", "company_size": "Medium",
        "location": "Germany", "remote_work": "Hybrid", "certifications": 1
    });

    let (status, one) = call(app.clone(), "POST", "/predict", profile.clone()).await;
    assert_eq!(status, 200);
    let s = one["salary"].as_f64().unwrap();
    assert!(one["low"].as_f64().unwrap() <= s && s <= one["high"].as_f64().unwrap());
    assert!((30_000.0..400_000.0).contains(&s));

    let (status, batch) = call(app.clone(), "POST", "/predict", json!([profile, profile])).await;
    assert_eq!(status, 200);
    assert_eq!(batch["predictions"].as_array().unwrap().len(), 2);
    assert_eq!(batch["predictions"][0], one);

    let mut bad = profile.clone();
    bad["location"] = json!("Mars");
    let (status, err) = call(app.clone(), "POST", "/predict", json!([profile, bad])).await;
    assert_eq!(status, 422);
    assert_eq!(err["index"], 1);

    let (status, info) = call(app.clone(), "GET", "/model", Value::Null).await;
    assert_eq!(status, 200);
    assert_eq!(info["forests"].as_array().unwrap().len(), 3);
    let (status, _) = call(app, "GET", "/health", Value::Null).await;
    assert_eq!(status, 200);
}
