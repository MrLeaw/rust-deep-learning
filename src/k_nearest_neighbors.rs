pub struct KNearestNeighbors {
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    k: usize,
}

impl KNearestNeighbors {
    pub fn new(x: Vec<Vec<f64>>, y: Vec<f64>, k: usize) -> KNearestNeighbors {
        KNearestNeighbors { x, y, k }
    }

    pub fn predict(&self, x: Vec<Vec<f64>>) -> Vec<f64> {
        x.iter().map(|x| self.predict_one(x.to_vec())).collect()
    }

    pub fn predict_one(&self, x: Vec<f64>) -> f64 {
        let mut distances = Vec::new();
        for i in 0..self.x.len() {
            let distance = euclidean_distance(&x, &self.x[i]);
            distances.push((distance, self.y[i]));
        }
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let mut sum = 0.0;
        for i in 0..self.k {
            sum += distances[i].1;
        }
        sum / self.k as f64
    }
}

fn euclidean_distance(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    let mut sum = 0.0;
    for i in 0..a.len() {
        sum += (a[i] - b[i]).powi(2);
    }
    sum.sqrt()
}

#[cfg(test)]
mod tests {
    use crate::datasets::{load_iris_dataset, train_test_split};

    use super::*;

    #[test]
    fn test_k_nearest_neighbors() {
        let (x, y) = load_iris_dataset();
        let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);

        let knn = KNearestNeighbors::new(x_train, y_train, 5);
        let predictions = knn.predict(x_test);
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
