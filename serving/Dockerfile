# Rust inference server. Build from the repository root, after exporting the models:
#   uv run python -m salary.export
#   docker build -f serving/Dockerfile -t salary-serve .
FROM rust:1-slim-bookworm AS build
WORKDIR /src
# Dependencies first, so code changes do not rebuild them.
COPY serving/Cargo.toml serving/Cargo.lock ./
RUN mkdir src && echo "fn main() {}" > src/main.rs && touch src/lib.rs \
    && cargo build --release --locked && rm -rf src
COPY serving/src ./src
RUN touch src/main.rs src/lib.rs && cargo build --release --locked

# Runtime: glibc + libgcc only, no shell, non-root user.
FROM gcr.io/distroless/cc-debian12:nonroot
COPY --from=build /src/target/release/salary-serve /usr/local/bin/salary-serve
COPY models/rust/schema.json models/rust/lightgbm.json models/rust/quantile_low.json models/rust/quantile_high.json /models/
ENV SALARY_MODEL_DIR=/models HOST=0.0.0.0 PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s CMD ["/usr/local/bin/salary-serve", "healthcheck"]
ENTRYPOINT ["/usr/local/bin/salary-serve"]
