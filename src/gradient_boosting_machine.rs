use std::collections::HashMap;

use crate::decision_tree::{DecisionTree, DecisionTreeSelectionCriteria};

// XGBoost
pub struct GradientBoostingMachine {
    n_estimators: usize,
    learning_rate: f64,
    min_impurity: f64,
    estimators: HashMap<usize, Vec<DecisionTree>>,
    classification_method: DecisionTreeSelectionCriteria,
    classes: Vec<usize>,
}

impl GradientBoostingMachine {
    pub fn new(
        n_estimators: usize,
        learning_rate: f64,
        min_impurity: f64,
        classification_method: DecisionTreeSelectionCriteria,
    ) -> Self {
        Self {
            n_estimators,
            learning_rate,
            min_impurity,
            estimators: HashMap::new(),
            classification_method,
            classes: vec![],
        }
    }

    pub fn fit(&mut self, x: &[Vec<f64>], y: &[usize]) {
        // Get unique classes
        let hashset = y
            .iter()
            .cloned()
            .collect::<std::collections::HashSet<usize>>();
        self.classes = hashset.into_iter().collect();
        println!("Classes: {:?}", self.classes);

        // Initialize estimators for each class
        for &class_label in &self.classes {
            self.estimators.insert(class_label, Vec::new());
        }

        // Initialize predictions for each class
        let mut y_pred = vec![vec![0.0; y.len()]; self.classes.len()]; // [num_classes][num_samples]

        for (class_idx, &class_label) in self.classes.iter().enumerate() {
            println!("Fitting class {}", class_label);
            // Gradient boosting iterations
            for estimator_index in 0..self.n_estimators {
                // Compute residuals for this class
                let residuals = compute_residuals_for_class(y, &y_pred[class_idx], class_label);
                // println!("Residuals: {:?}", residuals);

                // Train a tree on residuals
                let mut tree = DecisionTree::new(self.classification_method.clone());
                tree.fit(x, &residuals);

                // Predict with the new tree
                let tree_predictions = tree.predict(x);

                // Update predictions for this class
                for j in 0..y.len() {
                    y_pred[class_idx][j] += self.learning_rate * tree_predictions[j];
                }

                // Store the trained tree for this class
                self.estimators.get_mut(&class_label).unwrap().push(tree);

                println!("{:?}", y_pred[class_idx]);
            }

            // Check stopping condition (optional)
            let loss_value = self.loss_multiclass(y, &y_pred);
            println!("Loss value: {}", loss_value);
            if loss_value < self.min_impurity {
                break;
            }
        }
    }
    pub fn predict_one(&self, x: &[f64]) -> usize {
        let num_classes = self.classes.len();
        let mut y_pred = vec![0.0; num_classes];

        // Aggregate predictions from all estimators
        for (class_idx, &class_label) in self.classes.iter().enumerate() {
            for tree in self.estimators.get(&class_label).unwrap() {
                let tree_prediction = tree.predict_one(x);
                y_pred[class_idx] += self.learning_rate * tree_prediction;
            }
        }

        // Convert raw scores to probabilities using softmax
        let probabilities = softmax(&y_pred);

        // Return the class with the highest probability
        let max_index = probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        self.classes[max_index] // Map index to actual class label
    }

    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<usize> {
        x.iter().map(|sample| self.predict_one(sample)).collect()
    }

    fn loss_multiclass(&self, y_true: &[usize], y_pred: &[Vec<f64>]) -> f64 {
        assert_eq!(y_true.len(), y_pred[0].len());
        let mut loss = 0.0;
        for i in 0..y_true.len() {
            let class_idx = self.classes.iter().position(|&x| x == y_true[i]).unwrap();
            // debug message if y_pred is invalid (<= 0)
            if y_pred[class_idx][i] <= 0.0 {
                println!("Invalid prediction: {}", y_pred[class_idx][i]);
            }
            loss -= y_pred[class_idx][i].ln();
        }
        loss / y_true.len() as f64
    }
}

fn softmax(predictions: &[f64]) -> Vec<f64> {
    let max_pred = predictions
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let exp_preds: Vec<f64> = predictions.iter().map(|&p| (p - max_pred).exp()).collect();
    let sum_exp_preds: f64 = exp_preds.iter().sum();
    exp_preds.iter().map(|&p| p / sum_exp_preds).collect()
}

fn compute_residuals_for_class(y: &[usize], y_pred: &[f64], class_label: usize) -> Vec<f64> {
    y.iter()
        .enumerate()
        .map(|(i, &true_label)| {
            let true_prob = if true_label == class_label { 1.0 } else { 0.0 };
            true_prob - y_pred[i]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::datasets::{load_breast_cancer_dataset, load_iris_dataset, train_test_split};

    use super::*;

    #[test]
    fn test_gradient_boosting_machine() {
        let (x, y) = load_breast_cancer_dataset();
        let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);
        let y_train: Vec<usize> = y_train.iter().map(|&x| x as usize).collect();
        let y_test: Vec<usize> = y_test.iter().map(|&x| x as usize).collect();

        let mut gbm = GradientBoostingMachine::new(
            10,
            0.1,
            1e-7,
            DecisionTreeSelectionCriteria::VarianceReduction,
        );
        println!("Fitting...");
        gbm.fit(&x_train, &y_train);
        println!("Predicting...");
        let predictions = gbm.predict(&x_test);
        let mut correct = 0.0;
        for i in 0..y_test.len() {
            if predictions[i] == y_test[i] {
                correct += 1.0;
            }
        }
        let accuracy = correct / y_test.len() as f64;
        println!("Accuracy: {}", accuracy);
        assert!(accuracy > 0.9);
    }
}
