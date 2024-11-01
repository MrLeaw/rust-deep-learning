pub struct ConfusionMatrix {
    matrix: Vec<Vec<f64>>,
    classes: Vec<f64>,
}

impl ConfusionMatrix {
    pub fn print_matrix(&self) {
        for row in self.matrix.iter() {
            println!(
                "{}",
                row.iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<String>>()
                    .join(",")
            );
        }
    }

    pub fn accuracy(&self) -> f64 {
        let correct = self.diagonal().iter().sum::<f64>();
        let total = self
            .matrix
            .iter()
            .map(|row| row.iter().sum::<f64>())
            .sum::<f64>();
        correct / total
    }

    pub fn matrix(&self) -> &Vec<Vec<f64>> {
        &self.matrix
    }

    pub fn classes(&self) -> &Vec<f64> {
        &self.classes
    }

    pub fn true_positives(&self) -> Vec<f64> {
        self.matrix
            .iter()
            .enumerate()
            .map(|(i, _)| self.matrix[i][i])
            .collect()
    }

    pub fn false_positives(&self) -> Vec<f64> {
        self.matrix
            .iter()
            .enumerate()
            .map(|i| {
                self.matrix
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i.0)
                    .map(|(_, row)| row[i.0])
                    .sum::<f64>()
            })
            .collect()
    }

    pub fn true_negatives(&self) -> Vec<f64> {
        self.matrix
            .iter()
            .enumerate()
            .map(|i| {
                self.matrix
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i.0)
                    .map(|(_, row)| {
                        row.iter()
                            .enumerate()
                            .filter(|(k, _)| *k != i.0)
                            .map(|(_, x)| x)
                            .sum::<f64>()
                    })
                    .sum::<f64>()
            })
            .collect()
    }

    pub fn false_negatives(&self) -> Vec<f64> {
        self.matrix
            .iter()
            .enumerate()
            .map(|i| {
                self.matrix
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i.0)
                    .map(|(_, row)| row[i.0])
                    .sum::<f64>()
            })
            .collect()
    }

    pub fn tp_tn_fp_fn(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        (
            self.true_positives(),
            self.true_negatives(),
            self.false_positives(),
            self.false_negatives(),
        )
    }

    pub fn diagonal(&self) -> Vec<f64> {
        self.matrix
            .iter()
            .enumerate()
            .map(|(i, row)| row[i])
            .collect()
    }
    pub fn new(y_true: &Vec<f64>, y_pred: &Vec<f64>, classes: &Vec<f64>) -> Self {
        let mut matrix = vec![vec![0.0; classes.len()]; classes.len()];
        for (i, j) in y_true.iter().zip(y_pred.iter()) {
            matrix[classes.iter().position(|&x| x == *i).unwrap()][classes
                .iter()
                .position(|&x| x == *j)
                .expect(&format!("{} not in classes", j))] += 1.0;
        }
        Self {
            matrix,
            classes: classes.clone(),
        }
    }

    pub fn write_to_csv(&self, filename: &str) -> std::io::Result<()> {
        std::fs::write(
            filename,
            format!(
                "{}\n{}",
                self.classes
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<String>>()
                    .join(","),
                self.matrix
                    .iter()
                    .map(|row| {
                        row.iter()
                            .map(|x| x.to_string())
                            .collect::<Vec<String>>()
                            .join(",")
                    })
                    .collect::<Vec<String>>()
                    .join("\n"),
            ),
        )
    }
}
