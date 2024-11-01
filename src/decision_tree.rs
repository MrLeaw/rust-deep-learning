use std::{
    sync::{Arc, Mutex},
    thread,
};

use crate::utils::unique;

// enum for selecting method: information gain, gini impurity, etc.
#[derive(Clone)]
pub enum DecisionTreeClassificationMethod {
    InformationGain,
    GiniImpurity,
}

pub struct DecisionTree {
    root: DecisionTreeNode,
    selected_method: DecisionTreeClassificationMethod,
}

#[derive(Clone)]
struct DecisionTreeNode {
    left: Option<Box<DecisionTreeNode>>,
    right: Option<Box<DecisionTreeNode>>,
    best_feature: usize,
    best_threshold: f64,
    prediction: Option<f64>, // Holds the prediction for leaf nodes
}

impl DecisionTree {
    pub fn new(selected_method: DecisionTreeClassificationMethod) -> Self {
        Self {
            root: DecisionTreeNode {
                left: None,
                right: None,
                best_feature: 0,
                best_threshold: 0.0,
                prediction: None,
            },
            selected_method,
        }
    }

    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter().map(|x_i| self.predict_one(x_i)).collect()
    }

    fn predict_one(&self, x: &[f64]) -> f64 {
        let mut current_node = &self.root;
        while current_node.prediction.is_none() {
            let feature_value = x[current_node.best_feature];
            if feature_value < current_node.best_threshold {
                current_node = current_node.left.as_ref().unwrap();
            } else {
                current_node = current_node.right.as_ref().unwrap();
            }
        }
        current_node.prediction.unwrap()
    }

    pub fn fit_single_threaded(&mut self, x: &[Vec<f64>], y: &[f64]) {
        self.root = self.build_tree(x, y, true);
    }

    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) {
        self.root = self.build_tree(x, y, false);
    }

    fn build_tree(&self, x: &[Vec<f64>], y: &[f64], single_threaded: bool) -> DecisionTreeNode {
        if y.is_empty() {
            return DecisionTreeNode {
                left: None,
                right: None,
                best_feature: 0,
                best_threshold: 0.0,
                prediction: None,
            };
        }

        // Base case: if all labels are the same, return a leaf node with that label
        if y.iter().all(|&label| label == y[0]) {
            return DecisionTreeNode {
                left: None,
                right: None,
                best_feature: 0,
                best_threshold: 0.0,
                prediction: Some(y[0]),
            };
        }
        // If X is empty, return the most common label in y as a leaf prediction
        if x.is_empty() {
            let most_common_label = *y
                .iter()
                .max_by_key(|&&label| y.iter().filter(|&&val| val == label).count())
                .unwrap();
            return DecisionTreeNode {
                left: None,
                right: None,
                best_feature: 0,
                best_threshold: 0.0,
                prediction: Some(most_common_label),
            };
        }

        // Find the best feature and threshold to split on
        let best_feature;
        let best_threshold;
        match single_threaded {
            true => {
                let (bf, bt) = self.best_criteria_single_threaded(x, y);
                best_feature = bf;
                best_threshold = bt;
            }
            false => {
                let (bf, bt) = self.best_criteria(x, y);
                best_feature = bf;
                best_threshold = bt;
            }
        };

        // Split data based on the best feature and threshold
        let feature_column: Vec<f64> = x.iter().map(|row| row[best_feature]).collect();
        let (left_indices, right_indices) = split_indices(&feature_column, best_threshold);

        // Create subsets of x and y for left and right branches
        let left_x: Vec<Vec<f64>> = left_indices.iter().map(|&i| x[i].clone()).collect();
        let left_y: Vec<f64> = left_indices.iter().map(|&i| y[i]).collect();
        let right_x: Vec<Vec<f64>> = right_indices.iter().map(|&i| x[i].clone()).collect();
        let right_y: Vec<f64> = right_indices.iter().map(|&i| y[i]).collect();

        let left_node = self.build_tree(&left_x, &left_y, single_threaded);

        let right_node = self.build_tree(&right_x, &right_y, single_threaded);

        // Return the current node with its best splitting criteria and child nodes
        DecisionTreeNode {
            left: Some(Box::new(left_node)),
            right: Some(Box::new(right_node)),
            best_feature,
            best_threshold,
            prediction: None,
        }
    }

    fn best_criteria_single_threaded(&self, x: &[Vec<f64>], y: &[f64]) -> (usize, f64) {
        let mut best_score = 0.0;
        let mut best_feature = 0;
        let mut best_threshold = 0.0;
        for feature in 0..x[0].len() {
            let feature_column: Vec<f64> = x.iter().map(|row| row[feature]).collect();
            let mut thresholds = unique(&feature_column);
            thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            for &threshold in &thresholds {
                let score = match self.selected_method {
                    DecisionTreeClassificationMethod::InformationGain => {
                        information_gain(&y, &feature_column, threshold)
                    }
                    DecisionTreeClassificationMethod::GiniImpurity => {
                        gini_score(&y, &feature_column, threshold)
                    }
                };
                if score > best_score {
                    best_score = score;
                    best_feature = feature;
                    best_threshold = threshold;
                }
            }
        }
        (best_feature, best_threshold)
    }

    fn best_criteria(&self, x: &[Vec<f64>], y: &[f64]) -> (usize, f64) {
        let best_score = Arc::new(Mutex::new(0.0));
        let best_feature = Arc::new(Mutex::new(0));
        let best_threshold = Arc::new(Mutex::new(0.0));
        let x_arc = Arc::new(x.to_vec());
        let y_arc = Arc::new(y.to_vec());
        let mut handles = vec![];
        for feature in 0..x[0].len() {
            let x = Arc::clone(&x_arc);
            let y = Arc::clone(&y_arc);
            let best_score = Arc::clone(&best_score);
            let best_feature = Arc::clone(&best_feature);
            let best_threshold = Arc::clone(&best_threshold);
            let selected_method = self.selected_method.clone();
            let handle = thread::spawn(move || {
                let feature_column: Vec<f64> = x.iter().map(|row| row[feature]).collect();
                let mut thresholds = unique(&feature_column);
                thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                for &threshold in &thresholds {
                    let score = match selected_method {
                        DecisionTreeClassificationMethod::InformationGain => {
                            information_gain(&y, &feature_column, threshold)
                        }
                        DecisionTreeClassificationMethod::GiniImpurity => {
                            gini_score(&y, &feature_column, threshold)
                        }
                    };
                    let mut best_score_guard = best_score.lock().unwrap();
                    if score > *best_score_guard {
                        *best_score_guard = score;
                        *best_feature.lock().unwrap() = feature;
                        *best_threshold.lock().unwrap() = threshold;
                    }
                }
            });
            handles.push(handle);
        }
        for handle in handles {
            handle.join().unwrap();
        }
        let best_feature: usize = Arc::try_unwrap(best_feature).unwrap().into_inner().unwrap();
        let best_threshold = Arc::try_unwrap(best_threshold)
            .unwrap()
            .into_inner()
            .unwrap();
        (best_feature, best_threshold)
    }
}

fn entropy(y: &[f64]) -> f64 {
    let mut entropy = 0.0;
    let n = y.len() as f64;
    let classes = unique(y);
    for class in classes {
        let matches = y.iter().filter(|&&y_i| y_i == class);
        let sum = matches.count() as f64;
        let p: f64 = sum / n;
        entropy += p * p.log2();
    }
    -entropy
}

fn split(x_column: &[f64], threshold: f64) -> (Vec<f64>, Vec<f64>) {
    let mut left = vec![];
    let mut right = vec![];
    for &x in x_column {
        if x < threshold {
            left.push(x);
        } else {
            right.push(x);
        }
    }
    (left, right)
}

fn split_indices(x_column: &[f64], threshold: f64) -> (Vec<usize>, Vec<usize>) {
    let mut left = vec![];
    let mut right = vec![];
    for (i, &x) in x_column.iter().enumerate() {
        if x < threshold {
            left.push(i);
        } else {
            right.push(i);
        }
    }
    (left, right)
}
fn gini_score(y: &[f64], x_column: &[f64], threshold: f64) -> f64 {
    let parent_gini = gini(y);
    let (left, right) = split(x_column, threshold);
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }
    let n = y.len() as f64;
    let left_gini = gini(&left);
    let right_gini = gini(&right);
    let child_gini = (left.len() as f64 / n) * left_gini + (right.len() as f64 / n) * right_gini;
    parent_gini - child_gini
}

fn information_gain(y: &[f64], x_column: &[f64], threshold: f64) -> f64 {
    let parent_entropy = entropy(y);
    let (left_indices, right_indices) = split_indices(x_column, threshold);
    if left_indices.is_empty() || right_indices.is_empty() {
        return 0.0;
    }
    let n = y.len() as f64;
    let mut y_left = vec![];
    let mut y_right = vec![];
    for &i in &left_indices {
        y_left.push(y[i]);
    }
    for &i in &right_indices {
        y_right.push(y[i]);
    }
    let left_entropy = entropy(&y_left);
    let right_entropy = entropy(&y_right);
    let child_entropy =
        (y_left.len() as f64 / n) * left_entropy + (y_right.len() as f64 / n) * right_entropy;
    parent_entropy - child_entropy
}

fn gini(y: &[f64]) -> f64 {
    let mut gini = 1.0;
    let n = y.len() as f64;
    let classes = unique(y);
    for class in classes {
        let p = y.iter().filter(|&&y_i| y_i == class).sum::<f64>() / n;
        gini -= p * p;
    }
    gini
}

#[cfg(test)]
mod tests {
    use crate::{
        confusion_matrix::ConfusionMatrix,
        datasets::{load_breast_cancer_dataset, train_test_split},
    };

    use super::*;

    #[test]
    fn test_entropy() {
        let data = [
            0.05263, 0.04362, 0.0, 0.0, 0.1587, 0.05884, 0.3857, 1.428, 0.007189, 0.00466, 0.0,
            0.0, 0.02676, 0.002783, 0.08996, 0.06444, 0.0, 0.0, 0.2871, 0.07039,
        ];
        let entr = entropy(&data);
    }
    #[test]
    fn test_information_gain() {
        let X = vec![
            vec![2.0, 3.0],
            vec![1.0, 5.0],
            vec![3.0, 4.0],
            vec![6.0, 7.0],
        ];
        let y = vec![0.0, 0.0, 1.0, 1.0];
        let score = information_gain(&y, &X[0], 2.5);
        assert_eq!(score, 1.0);
    }

    #[test]
    fn test_information_gain_real_data() {
        let (x, y) = load_breast_cancer_dataset();
        for val in x {
            let score = information_gain(&y, &val, 2.5);
        }
    }

    #[test]
    fn test_decision_tree() {
        let (x, y) = load_breast_cancer_dataset();
        let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);
        let mut decision_tree =
            DecisionTree::new(DecisionTreeClassificationMethod::InformationGain);
        decision_tree.fit(&x_train, &y_train);

        let y_pred = decision_tree.predict(&x_test);
        let matrix = ConfusionMatrix::new(&y_test, &y_pred, &vec![0.0, 1.0]);
        matrix.print_matrix();
        assert!(matrix.accuracy() > 0.8);
    }

    #[test]
    fn test_best_criteria_real_data() {
        let (x, y) = load_breast_cancer_dataset();
        let tree = DecisionTree::new(DecisionTreeClassificationMethod::InformationGain);
        let (best_feature, best_threshold) = tree.best_criteria(&x, &y);
        assert_eq!(best_feature, 22);
        assert_eq!(best_threshold, 106.0);
    }
}
