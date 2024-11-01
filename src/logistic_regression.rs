use crate::utils::dot_product;

pub struct LogisticRegression {
    weights: Vec<f64>,
    bias: f64,
}

impl LogisticRegression {
    pub fn new() -> Self {
        Self {
            weights: vec![],
            bias: 0.0,
        }
    }

    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64], learning_rate: f64, iterations: usize) {
        self.weights = vec![0.0; x[0].len()];
        for _ in 0..iterations {
            for (x_i, y_i) in x.iter().zip(y.iter()) {
                let y_hat = sigmoid(dot_product(&self.weights, x_i) + self.bias);
                let error = y_i - y_hat;
                self.weights = self
                    .weights
                    .iter()
                    .zip(x_i.iter())
                    .map(|(w, x)| w + learning_rate * error * x)
                    .collect();
                self.bias += learning_rate * error;
            }
        }
    }

    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        // we need to make sure that very small values are rounded to 0 or 1, otherwise we cannot
        // use them as classes (for confusion matrix, ...)
        x.iter()
            .map(|x_i| {
                let y_hat = sigmoid(dot_product(&self.weights, x_i) + self.bias);
                if y_hat < 0.5 {
                    0.0
                } else {
                    1.0
                }
            })
            .collect()
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use crate::{
        confusion_matrix::ConfusionMatrix,
        datasets::{load_breast_cancer_dataset, train_test_split},
    };

    use super::*;
    #[test]
    fn test_logistic_regression() {
        let x = vec![vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]];
        let y = vec![0.0, 1.0, 1.0];
        let mut logistic_regression = LogisticRegression::new();
        logistic_regression.fit(&x, &y, 0.01, 1000);
        let y_pred = logistic_regression
            .predict(vec![vec![4.0, 5.0], vec![5.0, 6.0], vec![6.0, 7.0]].as_ref());
        let y_test = vec![1.0, 1.0, 1.0];
        for (index, value) in y_pred.iter().enumerate() {
            assert!((value - y_test[index]).abs() < 0.01);
        }
    }

    #[test]
    fn test_logistic_regression_2() {
        let (x, y) = load_breast_cancer_dataset();
        let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);
        let mut logistic_regression = LogisticRegression::new();
        logistic_regression.fit(&x_train, &y_train, 0.01, 1000);
        let y_pred = logistic_regression.predict(&x_test);
        println!("{:?}", y_pred);
        println!("{:?}", y_test);
        let matrix = ConfusionMatrix::new(&y_test, &y_pred, &vec![0.0, 1.0]);
        matrix.print_matrix();
    }
}
