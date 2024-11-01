pub fn unique(items: &[f64]) -> Vec<f64> {
    let mut result: Vec<f64> = vec![];
    for item in items {
        if !result.contains(item) {
            result.push(*item);
        }
    }
    result
}

pub fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn transpose(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut result = vec![vec![0.0; matrix.len()]; matrix[0].len()];
    for (i, row) in matrix.iter().enumerate() {
        for (j, &value) in row.iter().enumerate() {
            result[j][i] = value;
        }
    }
    result
}

pub fn matrix_multiply(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut result = vec![vec![0.0; b[0].len()]; a.len()];
    for i in 0..a.len() {
        for j in 0..b[0].len() {
            for k in 0..b.len() {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    result
}

pub fn identity_matrix(size: usize) -> Vec<Vec<f64>> {
    let mut result = vec![vec![0.0; size]; size];
    for i in 0..size {
        result[i][i] = 1.0;
    }
    result
}
pub fn invert_matrix(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = matrix.len();
    let mut augmented = matrix.to_vec();
    for i in 0..n {
        augmented[i].extend_from_slice(&identity_matrix(n)[i]);
    }

    for i in 0..n {
        // Check for zero pivot and handle it
        let pivot = augmented[i][i];
        if (pivot - 0.0).abs() < 1e-10 {
            // Small epsilon to avoid division by zero
            panic!("Matrix is singular or nearly singular; cannot invert.");
        }

        // Normalize the pivot row
        for j in 0..2 * n {
            augmented[i][j] /= pivot;
        }

        // Eliminate other rows
        for k in 0..n {
            if k != i {
                let factor = augmented[k][i];
                for j in 0..2 * n {
                    augmented[k][j] -= factor * augmented[i][j];
                }
            }
        }
    }

    augmented.iter().map(|row| row[n..].to_vec()).collect()
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        assert_eq!(dot_product(&a, &b), 32.0);
    }

    #[test]
    fn test_transpose() {
        let matrix = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let result = vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]];
        assert_eq!(transpose(&matrix), result);
    }

    #[test]
    fn test_matrix_multiply() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
        let result = vec![vec![19.0, 22.0], vec![43.0, 50.0]];
        assert_eq!(matrix_multiply(&a, &b), result);
    }

    #[test]
    fn test_invert_matrix() {
        let matrix = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let result = vec![vec![-2.0, 1.0], vec![1.5, -0.5]];
        assert_eq!(invert_matrix(&matrix), result);
    }

    #[test]
    fn test_identity_matrix() {
        let result = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        assert_eq!(identity_matrix(3), result);
    }
}
