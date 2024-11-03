// Support Vector Machine Algorithm

pub struct SupportVectorMachine {
    weights: Vec<f64>,
    bias: f64,
    c: f64,
    tol: f64,
    max_iter: u32,
    kernel: Kernel,
}

impl SupportVectorMachine {
    fn fit(&mut self, x: Vec<Vec<f64>>, y: Vec<f64>) {
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

    // Update alphas for each pair
    fn update_alphas(
        &mut self,
        i: usize,
        j: usize,
        error_i: f64,
        error_j: f64,
        x: &[Vec<f64>],
        y: &[f64],
        alpha: &mut [f64],
    ) {
        if i == j {
            return;
        }

        let c = self.c;
        let kernel = |a: &[f64], b: &[f64]| self.kernel(a, b);

        // Step 1: Calculate bounds L and H for alpha[j]
        let (l, h) = if y[i] != y[j] {
            (
                f64::max(0.0, alpha[j] - alpha[i]),
                f64::min(c, c + alpha[j] - alpha[i]),
            )
        } else {
            (
                f64::max(0.0, alpha[i] + alpha[j] - c),
                f64::min(c, alpha[i] + alpha[j]),
            )
        };

        if l == h {
            return;
        }

        // Step 2: Calculate eta (the optimal step size)
        let eta = kernel(&x[i], &x[i]) + kernel(&x[j], &x[j]) - 2.0 * kernel(&x[i], &x[j]);
        if eta <= 0.0 {
            return;
        }

        //   println!("Alpha[{}]: {}, Alpha[{}]: {}", i, alpha[i], j, alpha[j]);
        //   println!("Error_i: {}, Error_j: {}", error_i, error_j);
        //   println!("L: {}, H: {}", l, h);
        //   println!("Eta: {}", eta);

        // Step 3: Update alpha[j]
        let mut new_alpha_j = alpha[j] + y[j] * (error_i - error_j) / eta;
        // Clip new_alpha_j to be within bounds [L, H]
        new_alpha_j = new_alpha_j.max(l).min(h);

        // Step 4: Update alpha[i] based on new_alpha_j
        let delta_alpha_j = new_alpha_j - alpha[j];
        let new_alpha_i = alpha[i] - y[i] * y[j] * delta_alpha_j;

        alpha[i] = new_alpha_i;
        alpha[j] = new_alpha_j;

        // Step 5: Update the bias term
        let b1 = self.bias
            - error_i
            - y[i] * (alpha[i] - new_alpha_i) * kernel(&x[i], &x[i])
            - y[j] * (alpha[j] - new_alpha_j) * kernel(&x[i], &x[j]);

        let b2 = self.bias
            - error_j
            - y[i] * (alpha[i] - new_alpha_i) * kernel(&x[i], &x[j])
            - y[j] * (alpha[j] - new_alpha_j) * kernel(&x[j], &x[j]);

        // Choose the appropriate b value
        if 0.0 < new_alpha_i && new_alpha_i < c {
            self.bias = b1;
        } else if 0.0 < new_alpha_j && new_alpha_j < c {
            self.bias = b2;
        } else {
            self.bias = (b1 + b2) / 2.0;
        }
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

enum Kernel {
    Linear,
    Polynomial { degree: u32, coef: f64 },
    RBF { gamma: f64 },
    Sigmoid { coef: f64 },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::{load_breast_cancer_dataset, load_iris_dataset, train_test_split};

    //    #[test]
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
        };
        model.fit(x_train, y_train);
        let y_pred = model.predict(&x_test);
        for i in 0..y_test.len() {
            println!("Expected: {}, Predicted: {}", y_test[i], y_pred[i]);
        }
    }
}
