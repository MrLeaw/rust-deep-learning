use crate::decision_tree::{DecisionTree, DecisionTreeSelectionCriteria};
use rand::seq::SliceRandom;

pub struct RandomForest {
    trees: Vec<DecisionTree>,
    n_trees: usize,
    classification_method: DecisionTreeSelectionCriteria,
}

impl RandomForest {
    pub fn new(n_trees: usize, classification_method: DecisionTreeSelectionCriteria) -> Self {
        Self {
            trees: vec![],
            n_trees,
            classification_method,
        }
    }

    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) {
        for _ in 0..self.n_trees {
            let mut indices: Vec<usize> = (0..x.len()).collect();
            indices.shuffle(&mut rand::thread_rng());
            let x_train: Vec<Vec<f64>> = indices.iter().map(|&i| x[i].clone()).collect();
            let y_train: Vec<f64> = indices.iter().map(|&i| y[i]).collect();
            let mut tree = DecisionTree::new(self.classification_method.clone());
            tree.fit(&x_train, &y_train);
            self.trees.push(tree);
        }
    }

    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        let mut predictions = vec![0.0; x.len()];
        for tree in &self.trees {
            let tree_predictions = tree.predict(x);
            for i in 0..x.len() {
                predictions[i] += tree_predictions[i];
            }
        }
        predictions
            .iter()
            .map(|&x| x / self.n_trees as f64)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::datasets::{load_iris_dataset, train_test_split};

    use super::*;

    #[test]
    fn test_random_forest() {
        let (x, y) = load_iris_dataset();
        let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);

        let mut rf = RandomForest::new(10, DecisionTreeSelectionCriteria::InformationGain);
        rf.fit(&x_train, &y_train);
        let predictions = rf.predict(&x_test);
        let mut correct = 0.0;
        for i in 0..y_test.len() {
            if (predictions[i] - y_test[i]).abs() < 1e-6 {
                correct += 1.0
            }
        }
        println!("Accuracy: {}", correct / y_test.len() as f64);
        assert!(correct >= 0.7 * y_test.len() as f64);
    }
}
