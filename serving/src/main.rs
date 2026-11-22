//! `salary-serve` : serveur HTTP d'inférence.
//!
//!     SALARY_MODEL_DIR=models/rust HOST=127.0.0.1 PORT=8000 salary-serve
//!     salary-serve healthcheck        # pour le HEALTHCHECK Docker (image sans curl)

use std::{
    env,
    io::{Read, Write},
    net::TcpStream,
    path::PathBuf,
    process::ExitCode,
    sync::Arc,
    time::{Duration, Instant},
};

use salary_serve::{Engine, router};

fn port() -> u16 {
    env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8000)
}

/// Minimal HTTP GET /health on localhost, without any extra dependency.
fn healthcheck() -> ExitCode {
    let check = || -> std::io::Result<bool> {
        let mut s = TcpStream::connect(("127.0.0.1", port()))?;
        s.set_read_timeout(Some(Duration::from_secs(3)))?;
        s.write_all(b"GET /health HTTP/1.0\r\nHost: localhost\r\n\r\n")?;
        let mut buf = String::new();
        s.read_to_string(&mut buf)?;
        Ok(buf.starts_with("HTTP/1.0 200") || buf.starts_with("HTTP/1.1 200"))
    };
    if check().unwrap_or(false) {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

async fn shutdown() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let term = async {
        if let Ok(mut s) = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        {
            s.recv().await;
        }
    };
    #[cfg(not(unix))]
    let term = std::future::pending::<()>();
    tokio::select! { _ = ctrl_c => {}, _ = term => {} }
}

#[tokio::main]
async fn main() -> ExitCode {
    if env::args().nth(1).as_deref() == Some("healthcheck") {
        return healthcheck();
    }
    let dir = PathBuf::from(env::var("SALARY_MODEL_DIR").unwrap_or_else(|_| "models/rust".into()));
    let t0 = Instant::now();
    let engine = match Engine::load(&dir) {
        Ok(e) => Arc::new(e),
        Err(e) => {
            eprintln!(
                "Impossible de charger les modèles : {e}\nLancez `uv run python -m salary.export`."
            );
            return ExitCode::FAILURE;
        }
    };
    let trees: usize = [&engine.point, &engine.low, &engine.high]
        .iter()
        .map(|f| f.n_trees())
        .sum();
    let bytes: usize = [&engine.point, &engine.low, &engine.high]
        .iter()
        .map(|f| f.heap_bytes())
        .sum();
    let host = env::var("HOST").unwrap_or_else(|_| "127.0.0.1".into());
    let addr = format!("{host}:{}", port());
    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("Impossible d'écouter sur {addr} : {e}");
            return ExitCode::FAILURE;
        }
    };
    eprintln!(
        "salary-serve : {trees} arbres ({} Kio) chargés en {:.0?} depuis {}, écoute sur http://{addr}",
        bytes / 1024,
        t0.elapsed(),
        dir.display()
    );
    match axum::serve(listener, router(engine))
        .with_graceful_shutdown(shutdown())
        .await
    {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("Erreur du serveur : {e}");
            ExitCode::FAILURE
        }
    }
}
