use rand::seq::SliceRandom;
// for multiclass classification
pub fn load_iris_dataset() -> (Vec<Vec<f64>>, Vec<f64>) {
    let filepath = "data/iris";
    let x = std::fs::read_to_string(format!("{}/X.csv", filepath))
        .unwrap()
        .lines()
        .map(|line| line.split(',').map(|x| x.parse::<f64>().unwrap()).collect())
        .collect();
    let y = std::fs::read_to_string(format!("{}/y.csv", filepath))
        .unwrap()
        .lines()
        .map(|x| x.parse::<f64>().unwrap())
        .collect();
    (x, y)
}

// for binary classification
pub fn load_breast_cancer_dataset() -> (Vec<Vec<f64>>, Vec<f64>) {
    let filepath = "data/breast-cancer";
    let x = std::fs::read_to_string(format!("{}/X.csv", filepath))
        .unwrap()
        .lines()
        .map(|line| line.split(',').map(|x| x.parse::<f64>().unwrap()).collect())
        .collect();
    let y = std::fs::read_to_string(format!("{}/y.csv", filepath))
        .unwrap()
        .lines()
        .map(|x| x.parse::<f64>().unwrap())
        .collect();
    (x, y)
}

pub fn load_boston_dataset() -> (Vec<Vec<f64>>, Vec<f64>) {
    let filepath = "data/boston";
    // skip first line each because it's header
    let x = std::fs::read_to_string(format!("{}/X.csv", filepath))
        .unwrap()
        .lines()
        .skip(1)
        .map(|line| line.split(',').map(|x| x.parse::<f64>().unwrap()).collect())
        .collect();
    // one inner vec for each row
    let y = std::fs::read_to_string(format!("{}/y.csv", filepath))
        .unwrap()
        .lines()
        .skip(1)
        .map(|x| x.parse::<f64>().unwrap())
        .collect();
    (x, y)
}

// for regression
pub fn load_diabetes_dataset() -> (Vec<Vec<f64>>, Vec<f64>) {
    let filepath = "data/diabetes";
    let x = std::fs::read_to_string(format!("{}/X.csv", filepath))
        .unwrap()
        .lines()
        .map(|line| line.split(',').map(|x| x.parse::<f64>().unwrap()).collect())
        .collect();
    let y = std::fs::read_to_string(format!("{}/y.csv", filepath))
        .unwrap()
        .lines()
        .map(|x| x.parse::<f64>().unwrap())
        .collect();
    (x, y)
}

pub fn train_test_split(
    x: &[Vec<f64>],
    y: &[f64],
    test_percent: f64,
    random: bool,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    let mut x = x.to_vec();
    let mut y = y.to_vec();
    if random {
        let mut rng = rand::thread_rng();
        let mut indices: Vec<usize> = (0..x.len()).collect();
        indices.shuffle(&mut rng);
        x = indices.iter().map(|i| x[*i].clone()).collect();
        y = indices.iter().map(|i| y[*i].clone()).collect();
    }
    let test_size = (x.len() as f64 * test_percent) as usize;
    let x_train = x[test_size..].to_vec();
    let x_test = x[..test_size].to_vec();
    let y_train = y[test_size..].to_vec();
    let y_test = y[..test_size].to_vec();
    (x_train, x_test, y_train, y_test)
}
