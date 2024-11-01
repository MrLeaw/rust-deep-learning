use crate::utils::{dot_product, invert_matrix, matrix_multiply, transpose};

pub struct LinearRegression {
    intercept: f64,
    coefficients: Vec<f64>,
}

impl LinearRegression {
    pub fn new() -> Self {
        Self {
            intercept: 0.0,
            coefficients: vec![],
        }
    }

    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) {
        let x_with_bias: Vec<Vec<f64>> = x
            .iter()
            .map(|row| {
                let mut new_row = vec![1.0];
                new_row.extend(row);
                new_row
            })
            .collect();

        let x_transpose = transpose(&x_with_bias);
        let xtx = matrix_multiply(&x_transpose, &x_with_bias);
        let xtx_inv = invert_matrix(&xtx);
        let xty = matrix_multiply(&x_transpose, &transpose(&[y.to_vec()]));
        let coefficients = matrix_multiply(&xtx_inv, &xty);
        self.intercept = coefficients[0][0];
        self.coefficients = coefficients[1..].iter().map(|row| row[0]).collect();
    }

    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter()
            .map(|x_i| self.intercept + dot_product(&self.coefficients, x_i.as_slice()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[should_panic]
    fn test_linear_regression() {
        let x = vec![vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]];
        let y = vec![3.0, 5.0, 7.0];
        let mut linear_regression = LinearRegression::new();
        linear_regression.fit(&x, &y);
        let _ = linear_regression
            .predict(vec![vec![4.0, 5.0], vec![5.0, 6.0], vec![6.0, 7.0]].as_ref());
    }

    #[test]
    fn test_linear_regression_non_singular() {
        let x = vec![vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 5.0]];
        let y = vec![4.0, 6.0, 8.0];
        let mut linear_regression = LinearRegression::new();

        linear_regression.fit(&x, &y);

        let y_pred =
            linear_regression.predict(&vec![vec![4.0, 7.0], vec![5.0, 8.0], vec![6.0, 9.0]]);

        let y_test = vec![10.0, 12.0, 14.0];
        for (index, value) in y_pred.iter().enumerate() {
            println!("Predicted: {}, Expected: {}", value, y_test[index]);
            assert!((value - y_test[index]).abs() < 0.01);
        }
    }
}
