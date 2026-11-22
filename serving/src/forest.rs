//! Gradient-boosted forest evaluation (bundle format `salary-forest/1`, see `salary.export`).
//!
//! All nodes of all trees live in one contiguous array of 24-byte nodes, so the whole
//! model fits in a few hundred kilobytes and evaluation walks memory linearly.

use std::{fs, mem::size_of, path::Path};

use serde::Deserialize;

pub const FORMAT: &str = "salary-forest/1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Transform {
    Identity,
    Exp,
}

#[derive(Deserialize)]
struct ForestFile {
    format: String,
    name: String,
    objective: String,
    transform: Transform,
    n_features: usize,
    roots: Vec<i32>,
    split_feature: Vec<u32>,
    threshold: Vec<f64>,
    left: Vec<i32>,
    right: Vec<i32>,
    leaf_value: Vec<f64>,
}

/// Internal node: go to `children[0]` when `x[feature] <= threshold`, else `children[1]`.
/// A negative child `c` is leaf `!c`. Indexing by the comparison keeps the walk branchless.
#[derive(Debug, Clone, Copy)]
struct Node {
    threshold: f64,
    feature: u32,
    children: [i32; 2],
}

#[derive(Debug)]
pub struct Forest {
    pub name: String,
    pub objective: String,
    pub transform: Transform,
    n_features: usize,
    roots: Box<[i32]>,
    nodes: Box<[Node]>,
    leaves: Box<[f64]>,
}

impl Forest {
    pub fn load(path: &Path) -> Result<Self, String> {
        let text = fs::read_to_string(path).map_err(|e| format!("{} : {e}", path.display()))?;
        let f: ForestFile = serde_json::from_str(&text)
            .map_err(|e| format!("{} : JSON invalide : {e}", path.display()))?;
        Self::from_file(f).map_err(|e| format!("{} : {e}", path.display()))
    }

    fn from_file(f: ForestFile) -> Result<Self, String> {
        if f.format != FORMAT {
            return Err(format!("format {} inattendu (attendu {FORMAT})", f.format));
        }
        let n = f.split_feature.len();
        if [f.threshold.len(), f.left.len(), f.right.len()] != [n, n, n] {
            return Err("tableaux de nœuds de tailles différentes".into());
        }
        let n_leaves = f.leaf_value.len() as i64;
        let valid = |c: i32| (c >= 0 && (c as usize) < n) || (c < 0 && i64::from(!c) < n_leaves);
        if !f
            .roots
            .iter()
            .chain(&f.left)
            .chain(&f.right)
            .all(|&c| valid(c))
        {
            return Err("indice d'enfant hors limites".into());
        }
        if f.split_feature.iter().any(|&j| j as usize >= f.n_features) {
            return Err("indice de variable hors limites".into());
        }
        // A child index always points forward (pre-order), so no cycle is possible.
        if (0..n).any(|i| {
            [f.left[i], f.right[i]]
                .iter()
                .any(|&c| c >= 0 && c as usize <= i)
        }) {
            return Err("arbre mal ordonné".into());
        }
        let nodes = (0..n)
            .map(|i| Node {
                threshold: f.threshold[i],
                feature: f.split_feature[i],
                children: [f.left[i], f.right[i]],
            })
            .collect();
        Ok(Self {
            name: f.name,
            objective: f.objective,
            transform: f.transform,
            n_features: f.n_features,
            roots: f.roots.into_boxed_slice(),
            nodes,
            leaves: f.leaf_value.into_boxed_slice(),
        })
    }

    #[inline(always)]
    fn leaf(&self, root: i32, x: &[f64]) -> f64 {
        let mut i = root;
        while i >= 0 {
            let n = &self.nodes[i as usize];
            i = n.children[(x[n.feature as usize] > n.threshold) as usize];
        }
        self.leaves[!i as usize]
    }

    /// Sum of one leaf per tree, in tree order (same order as LightGBM).
    pub fn predict_raw(&self, x: &[f64]) -> f64 {
        assert_eq!(x.len(), self.n_features, "nombre de variables");
        self.roots.iter().map(|&root| self.leaf(root, x)).sum()
    }

    /// Raw scores for `rows` (row-major, `n_features` values per row), added to `out`.
    ///
    /// Trees are the outer loop, so each tree stays in L1 cache while every row walks it.
    /// Each row still sums its leaves in tree order: results are bit-identical to `predict_raw`.
    pub fn add_raw_batch(&self, rows: &[f64], out: &mut [f64]) {
        let nf = self.n_features;
        assert_eq!(rows.len(), out.len() * nf, "nombre de variables");
        for &root in self.roots.iter() {
            for (x, o) in rows.chunks_exact(nf).zip(out.iter_mut()) {
                *o += self.leaf(root, x);
            }
        }
    }

    pub fn finish(&self, raw: f64) -> f64 {
        match self.transform {
            Transform::Identity => raw,
            Transform::Exp => raw.exp(),
        }
    }

    pub fn predict(&self, x: &[f64]) -> f64 {
        self.finish(self.predict_raw(x))
    }

    pub fn n_features(&self) -> usize {
        self.n_features
    }

    pub fn n_trees(&self) -> usize {
        self.roots.len()
    }

    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Heap bytes used by the model arrays.
    pub fn heap_bytes(&self) -> usize {
        self.nodes.len() * size_of::<Node>()
            + self.leaves.len() * size_of::<f64>()
            + self.roots.len() * size_of::<i32>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two stumps on feature 0 and a single-leaf tree.
    fn tiny(transform: Transform) -> Forest {
        Forest::from_file(ForestFile {
            format: FORMAT.into(),
            name: "tiny".into(),
            objective: "regression".into(),
            transform,
            n_features: 2,
            roots: vec![0, 1, !4],
            split_feature: vec![0, 1],
            threshold: vec![1.5, 0.5],
            left: vec![!0, !2],
            right: vec![!1, !3],
            leaf_value: vec![1.0, 2.0, 10.0, 20.0, 0.5],
        })
        .unwrap()
    }

    #[test]
    fn walks_left_on_less_or_equal() {
        let f = tiny(Transform::Identity);
        assert_eq!(f.predict(&[1.5, 0.5]), 1.0 + 10.0 + 0.5);
        assert_eq!(f.predict(&[2.0, 1.0]), 2.0 + 20.0 + 0.5);
        assert_eq!(f.n_trees(), 3);
    }

    #[test]
    fn batch_equals_single() {
        let f = tiny(Transform::Identity);
        let rows = [1.5, 0.5, 2.0, 1.0, 0.0, 9.0];
        let mut out = [0.0; 3];
        f.add_raw_batch(&rows, &mut out);
        for (i, o) in out.iter().enumerate() {
            assert_eq!(f.predict_raw(&rows[2 * i..2 * i + 2]), *o);
        }
    }

    #[test]
    fn exp_transform() {
        let f = tiny(Transform::Exp);
        assert!((f.predict(&[0.0, 0.0]) - 11.5_f64.exp()).abs() < 1e-9);
    }

    #[test]
    fn rejects_bad_child() {
        let bad = ForestFile {
            format: FORMAT.into(),
            name: "bad".into(),
            objective: "regression".into(),
            transform: Transform::Identity,
            n_features: 1,
            roots: vec![0],
            split_feature: vec![0],
            threshold: vec![0.0],
            left: vec![0],
            right: vec![!0],
            leaf_value: vec![1.0],
        };
        assert!(Forest::from_file(bad).is_err());
    }
}
