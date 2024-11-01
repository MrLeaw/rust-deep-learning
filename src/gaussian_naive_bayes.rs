use std::{cmp::Ordering, f64::consts::PI};

use crate::utils::unique;

#[derive(Debug)]
pub struct GaussianNBClassifier {
    classes: Vec<f64>,
    priors: Vec<f64>,
    variances: Vec<Vec<f64>>,
    theta: Vec<Vec<f64>>,
    class_count: Vec<usize>,
}

impl GaussianNBClassifier {
    fn joint_log_likelihood(&self, x: &[f64]) -> Vec<f64> {
        let mut result = vec![];
        for (class_index, &_class_) in self.classes.iter().enumerate() {
            let jointi = self.priors[class_index].ln();
            let mut n_ij = -0.5 * (2.0 * PI * self.variances[class_index].len() as f64).ln();
            for (j, (variance, &x_i)) in
                self.variances[class_index].iter().zip(x.iter()).enumerate()
            {
                n_ij -=
                    0.5 * ((x_i - self.theta[class_index][j]).powi(2) / variance + variance.ln());
            }
            result.push(jointi + n_ij);
        }
        result
    }

    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter()
            .map(|x| {
                let joint_log_likelihood = self.joint_log_likelihood(x);
                let max_index = joint_log_likelihood
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                    .unwrap()
                    .0;
                self.classes[max_index]
            })
            .collect()
    }

    pub fn new(classes: Vec<f64>) -> Self {
        Self {
            classes,
            priors: vec![],
            theta: vec![],
            variances: vec![],
            class_count: vec![],
        }
    }
    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) {
        let n_features = x[0].len();
        let n_classes = self.classes.len();
        self.theta = vec![vec![0.0; n_features]; n_classes];
        self.variances = vec![vec![0.0; n_features]; n_classes];
        self.priors = vec![0.0; n_classes];
        self.class_count = vec![0; n_classes];
        let unique_y = unique(y);
        let unique_y_in_classes = self
            .classes
            .iter()
            .filter(|class_| unique_y.contains(class_))
            .collect::<Vec<_>>();
        // check if all values in y are in classes, if not panic
        if unique_y_in_classes.len() != unique_y.len() {
            panic!("All values in y must be in classes");
        }
        for y_i in unique_y {
            let class_index = self.classes.iter().position(|&x| x == y_i).unwrap();
            let x_class = x
                .iter()
                .zip(y.iter())
                .filter(|(_, &y)| y == y_i)
                .map(|(x, _)| x)
                .collect::<Vec<_>>();
            let n_samples = x_class.len();
            self.class_count[class_index] = n_samples;
            self.priors[class_index] = n_samples as f64 / x.len() as f64;
            for j in 0..n_features {
                let feature_values = x_class.iter().map(|x| x[j]);
                let mean = feature_values.clone().sum::<f64>() / n_samples as f64;
                let variance =
                    feature_values.map(|x| (x - mean).powi(2)).sum::<f64>() / n_samples as f64;
                self.theta[class_index][j] = mean;
                self.variances[class_index][j] = variance;
            }
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::{
        confusion_matrix::ConfusionMatrix,
        datasets::{load_iris_dataset, train_test_split},
    };

    use super::*;
    #[test]
    fn test_gnb() {
        let (x, y) = load_iris_dataset();
        let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);
        let classes = vec![0.0, 1.0, 2.0];
        let mut gnb = GaussianNBClassifier::new(classes.clone());
        gnb.fit(&x_train, &y_train);
        let y_pred = gnb.predict(&x_test);
        let matrix = ConfusionMatrix::new(&y_test, &y_pred, &classes);
        println!("Accuracy: {}", matrix.accuracy());
        assert!(matrix.accuracy() >= 0.9);
    }
}
