use crate::utils::dot_product;

pub struct StochasticGradientDescent {
    learning_rate: f64,
    iterations: usize,
    bias: f64,
    weights: Vec<f64>,
}
impl StochasticGradientDescent {
    pub fn new(learning_rate: f64, iterations: usize) -> Self {
        Self {
            learning_rate,
            iterations,
            bias: 0.0,
            weights: vec![],
        }
    }

    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) {
        self.weights = vec![0.0; x[0].len()];
        for _ in 0..self.iterations {
            for (x_i, y_i) in x.iter().zip(y.iter()) {
                let y_hat = dot_product(&self.weights, x_i) + self.bias;
                let error = y_i - y_hat;
                self.weights = self
                    .weights
                    .iter()
                    .zip(x_i.iter())
                    .map(|(w, x)| w + self.learning_rate * error * x)
                    .collect();
                self.bias += self.learning_rate * error;
            }
        }
    }

    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter()
            .map(|x_i| dot_product(&self.weights, x_i) + self.bias)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_sgd() {
        let x = vec![vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]];
        let y = vec![3.0, 5.0, 7.0];
        let mut sgd = StochasticGradientDescent::new(0.01, 1000);
        sgd.fit(&x, &y);
        let y_pred = sgd.predict(vec![vec![4.0, 5.0], vec![5.0, 6.0], vec![6.0, 7.0]].as_ref());
        let y_test = vec![9.0, 11.0, 13.0];
        for (index, value) in y_pred.iter().enumerate() {
            assert!((value - y_test[index]).abs() < 0.01);
        }
    }
}