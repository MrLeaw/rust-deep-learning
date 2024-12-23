// Support Vector Machine Algorithm

pub struct SupportVectorMachine {
    weights: Vec<f64>,
    bias: f64,
    c: f64,
    tol: f64,
    max_iter: u32,
    kernel: Kernel,
    threads: usize, // for parallel processing
}

impl SupportVectorMachine {
    pub fn new(c: f64, tol: f64, max_iter: u32, kernel: Kernel, threads: usize) -> Self {
        Self {
            weights: vec![],
            bias: 0.0,
            c,
            tol,
            max_iter,
            kernel,
            threads,
        }
    }

    pub fn fit(&mut self, x: Vec<Vec<f64>>, y: Vec<f64>) {
        // validate y values (only -1 and 1 are allowed)
        for yi in &y {
            if !matches!(yi, -1.0 | 1.0) {
                panic!(
                    "Only -1 and 1 are allowed as class labels. Found label: {}",
                    yi
                );
            }
        }

        let n_samples = x.len();
        let mut alpha = vec![0.0; n_samples];
        self.bias = 0.0;

        for i in 0..self.max_iter {
            println!("Iteration {}", i);
            let mut num_changed_alphas = 0;

            for i in 0..n_samples {
                let error_i = self.compute_error(&x, &y, &alpha, i);

                if (y[i] * error_i < -self.tol && alpha[i] < self.c)
                    || (y[i] * error_i > self.tol && alpha[i] > 0.0)
                {
                    let j = (i + 1) % n_samples; // Choose a second index j
                    let error_j = self.compute_error(&x, &y, &alpha, j);

                    // Update alphas and bias
                    self.update_alphas(i, j, error_i, error_j, &x, &y, &mut alpha);
                    num_changed_alphas += 1;
                }
            }
            println!(
                "alpha: {:?}, bias: {}, error: {}",
                alpha,
                self.bias,
                self.compute_error(&x, &y, &alpha, 0)
            );
            if num_changed_alphas == 0 {
                break;
            }
        }

        // After training, compute weights from alphas for linear kernel
        if matches!(self.kernel, Kernel::Linear) {
            self.weights = x
                .iter()
                .enumerate()
                .map(|(i, xi)| xi.iter().map(|xi_j| alpha[i] * y[i] * xi_j).collect())
                .fold(vec![0.0; x[0].len()], |acc, vec: Vec<f64>| {
                    acc.iter().zip(vec).map(|(a, b)| a + b).collect()
                });
        }
    }

    fn compute_error(&self, x: &[Vec<f64>], y: &[f64], alpha: &[f64], i: usize) -> f64 {
        let prediction: f64 = x
            .iter()
            .enumerate()
            .map(|(j, x_j)| alpha[j] * y[j] * self.kernel(&x[i], x_j))
            .sum::<f64>()
            + self.bias;
        prediction - y[i]
    }
    fn update_alphas(
        &mut self,
        i: usize,
        j: usize,
        error_i: f64,
        error_j: f64,
        x: &Vec<Vec<f64>>,
        y: &Vec<f64>,
        alpha: &mut Vec<f64>,
    ) {
        if i == j {
            return; // Avoid self-pairing
        }

        let (alpha_i_old, alpha_j_old) = (alpha[i], alpha[j]);

        // Compute the bounds (L and H) for alpha[j]
        let (low, high) = if y[i] != y[j] {
            (
                0.0f64.max(alpha[j] - alpha[i]),
                self.c.min(self.c + alpha[j] - alpha[i]),
            )
        } else {
            (
                0.0f64.max(alpha[i] + alpha[j] - self.c),
                self.c.min(alpha[i] + alpha[j]),
            )
        };

        if (high - low).abs() < 1e-5 {
            return; // Bounds are too close, skip this pair
        }

        // Compute eta (second derivative of the objective function)
        let eta =
            2.0 * self.kernel(&x[i], &x[j]) - self.kernel(&x[i], &x[i]) - self.kernel(&x[j], &x[j]);

        if eta >= 0.0 {
            return; // Not a suitable pair to optimize
        }

        // Update alpha[j]
        alpha[j] = alpha_j_old - y[j] * (error_i - error_j) / eta;

        // Clip alpha[j] to the bounds
        alpha[j] = alpha[j].max(low).min(high);

        if (alpha[j] - alpha_j_old).abs() < 1e-5 {
            return; // Change in alpha[j] is too small
        }

        // Update alpha[i]
        alpha[i] = alpha_i_old + y[i] * y[j] * (alpha_j_old - alpha[j]);

        // Compute the updated bias
        let b1 = self.bias
            - error_i
            - y[i] * (alpha[i] - alpha_i_old) * self.kernel(&x[i], &x[i])
            - y[j] * (alpha[j] - alpha_j_old) * self.kernel(&x[i], &x[j]);

        let b2 = self.bias
            - error_j
            - y[i] * (alpha[i] - alpha_i_old) * self.kernel(&x[i], &x[j])
            - y[j] * (alpha[j] - alpha_j_old) * self.kernel(&x[j], &x[j]);

        self.bias = if 0.0 < alpha[i] && alpha[i] < self.c {
            b1
        } else if 0.0 < alpha[j] && alpha[j] < self.c {
            b2
        } else {
            (b1 + b2) / 2.0
        };
    }
    fn kernel(&self, x_i: &[f64], x_j: &[f64]) -> f64 {
        match self.kernel {
            Kernel::Linear => x_i.iter().zip(x_j).map(|(a, b)| a * b).sum(),
            Kernel::Polynomial { degree, coef } => {
                (x_i.iter().zip(x_j).map(|(a, b)| a * b).sum::<f64>() + coef).powi(degree as i32)
            }
            Kernel::RBF { gamma } => {
                let squared_distance: f64 = x_i.iter().zip(x_j).map(|(a, b)| (a - b).powi(2)).sum();
                (-gamma * squared_distance).exp()
            }
            Kernel::Sigmoid { coef } => {
                (x_i.iter().zip(x_j).map(|(a, b)| a * b).sum::<f64>() + coef).tanh()
            }
        }
    }
    fn kernel_linear(x1: &Vec<f64>, x2: &Vec<f64>) -> f64 {
        let mut sum = 0.0;
        for i in 0..x1.len() {
            sum += x1[i] * x2[i];
        }
        sum
    }

    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        let mut results = Vec::new();
        for xi in x {
            results.push(self.predict_single(xi));
        }
        results
    }

    fn predict_single(&self, x: &[f64]) -> f64 {
        let result = match self.kernel {
            Kernel::Linear => {
                self.weights
                    .iter()
                    .zip(x)
                    .map(|(w, xi)| w * xi)
                    .sum::<f64>()
                    + self.bias
            }
            _ => {
                // Non-linear kernel requires summing over all support vectors
                // Use kernel function on each support vector and sum the results
                // Implement logic for non-linear case
                0.0
            }
        };
        if result >= 0.0 {
            1.0
        } else {
            -1.0
        }
    }
}

fn standardize(data: &mut Vec<Vec<f64>>) {
    let num_features = data[0].len();
    let means: Vec<f64> = (0..num_features)
        .map(|j| data.iter().map(|x| x[j]).sum::<f64>() / data.len() as f64)
        .collect();

    let std_devs: Vec<f64> = (0..num_features)
        .map(|j| {
            let mean = means[j];
            (data.iter().map(|x| (x[j] - mean).powi(2)).sum::<f64>() / data.len() as f64).sqrt()
        })
        .collect();

    for x in data.iter_mut() {
        for j in 0..num_features {
            if std_devs[j] != 0.0 {
                x[j] = (x[j] - means[j]) / std_devs[j];
            }
        }
    }
}

fn convert_y(y: Vec<f64>) -> Vec<f64> {
    y.iter()
        .map(|&yi| if yi == 0.0 { -1.0 } else { 1.0 })
        .collect()
}

pub enum Kernel {
    Linear,
    Polynomial { degree: u32, coef: f64 },
    RBF { gamma: f64 },
    Sigmoid { coef: f64 },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::{load_breast_cancer_dataset, load_iris_dataset, train_test_split};

    // test kernels
    #[test]
    fn test_kernel_linear() {
        let x1 = vec![1.0, 2.0, 3.0];
        let x2 = vec![4.0, 5.0, 6.0];
        let result = SupportVectorMachine::new(0.0, 0.0, 0, Kernel::Linear, 8).kernel(&x1, &x2);
        assert_eq!(result, 32.0); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_kernel_polynomial() {
        let x1 = vec![1.0, 2.0];
        let x2 = vec![3.0, 4.0];
        let degree = 2;
        let coef = 1.0;
        let result = SupportVectorMachine::new(0.0, 0.0, 0, Kernel::Polynomial { degree, coef }, 8)
            .kernel(&x1, &x2);
        let expected = ((1.0 * 3.0 + 2.0 * 4.0) + coef).powi(degree as i32);
        assert_eq!(result, expected); // (3 + 8 + 1)^2 = 144
    }

    #[test]
    fn test_kernel_rbf() {
        let x1 = vec![1.0, 0.0];
        let x2 = vec![0.0, 1.0];
        let gamma = 0.5;
        let result =
            SupportVectorMachine::new(0.0, 0.0, 0, Kernel::RBF { gamma }, 8).kernel(&x1, &x2);
        let squared_distance = 2.0;
        let expected = (-gamma * squared_distance).exp();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_kernel_sigmoid() {
        let x1 = vec![1.0, -1.0];
        let x2 = vec![1.0, 1.0];
        let coef = 0.0;
        let result =
            SupportVectorMachine::new(0.0, 0.0, 0, Kernel::Sigmoid { coef }, 8).kernel(&x1, &x2);
        let dot_product = 1.0 * 1.0 + (-1.0) * 1.0; // 1 - 1 = 0
        let expected = (dot_product + coef).tanh();
        assert_eq!(result, expected);
    }
    #[test]
    fn test_support_vector_machine() {
        let (mut x, y) = load_breast_cancer_dataset();
        standardize(&mut x);
        let y = convert_y(y);
        let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);

        let mut model = SupportVectorMachine {
            weights: vec![],
            bias: 0.0,
            c: 10.0,
            tol: 1e-4,
            max_iter: 200,
            kernel: Kernel::Linear,
            threads: 8,
        };
        model.fit(x_train, y_train);
        let y_pred = model.predict(&x_test);
        let accuracy = y_test
            .iter()
            .zip(y_pred.iter())
            .filter(|(&a, &b)| a == b)
            .count() as f64
            / y_test.len() as f64;
        println!("Accuracy: {}", accuracy);
        assert!(accuracy > 0.9);
    }
}
