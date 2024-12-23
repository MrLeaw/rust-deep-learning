use deep_learning::confusion_matrix::ConfusionMatrix;
use deep_learning::datasets::{
    load_breast_cancer_dataset, load_diabetes_dataset, load_iris_dataset, train_test_split,
};
use deep_learning::decision_tree::{DecisionTree, DecisionTreeClassificationMethod};
use deep_learning::gaussian_naive_bayes::GaussianNBClassifier;
use deep_learning::linear_regression::LinearRegression;
use deep_learning::logistic_regression::LogisticRegression;
use deep_learning::stochastic_gradient_descent::StochasticGradientDescent;
use deep_learning::support_vector_machine::{Kernel, SupportVectorMachine};

fn main() {
    let mut timings = Vec::new();
    // time the execution of each algorithm
    println!("Running stochastic gradient descent...");
    let now = std::time::Instant::now();
    run_stochastic_gd();
    timings.push(("SGD", (now.elapsed().as_nanos() as f64) / 1_000_000.0));
    println!("Time taken: {:?}\n", now.elapsed());
    println!("Running linear regression...");
    let now = std::time::Instant::now();
    run_linear_regression();
    timings.push(("Linear", (now.elapsed().as_nanos() as f64) / 1_000_000.0));
    println!("Time taken: {:?}\n", now.elapsed());
    println!("Running Gaussian Naive Bayes (Breast Cancer)...");
    let now = std::time::Instant::now();
    run_gaussian_nb();
    timings.push((
        "GaussianNB Breast Cancer",
        (now.elapsed().as_nanos() as f64) / 1_000_000.0,
    ));
    println!("Time taken: {:?}\n", now.elapsed());
    println!("Running Gaussian Naive Bayes Multi-class (Iris)...");
    let now = std::time::Instant::now();
    run_gaussian_nb_iris();
    timings.push((
        "GaussianNB Iris",
        (now.elapsed().as_nanos() as f64) / 1_000_000.0,
    ));
    println!("Time taken: {:?}\n", now.elapsed());
    println!("Running logistic regression (Breast Cancer)...");
    let now = std::time::Instant::now();
    run_logistic_regression();
    timings.push((
        "Logistic Regression Breast Cancer",
        (now.elapsed().as_nanos() as f64) / 1_000_000.0,
    ));
    println!(
        "Time taken: {:?}\n",
        (now.elapsed().as_nanos() as f64) / 1_000_000.0
    );
    println!("Running decision tree (Breast Cancer)...");
    let now = std::time::Instant::now();
    run_decision_tree();
    timings.push((
        "Decision Tree Breast Cancer",
        (now.elapsed().as_nanos() as f64) / 1_000_000.0,
    ));
    println!("Time taken: {:?}\n", now.elapsed());
    println!("Running decision tree Multi-class (Iris)...");
    let now = std::time::Instant::now();
    run_decision_tree_iris();
    timings.push((
        "Decision Tree Iris",
        (now.elapsed().as_nanos() as f64) / 1_000_000.0,
    ));
    println!("Time taken: {:?}\n", now.elapsed());

    println!("Running SVM (Breast Cancer)...");
    let now = std::time::Instant::now();
    run_svm();
    timings.push(("SVM", (now.elapsed().as_nanos() as f64) / 1_000_000.0));
    println!("Time taken: {:?}\n", now.elapsed());

    // write timings to a csv file
    let _ = std::fs::write(
        "timings.csv",
        format!(
            "Model,Time (ms)\n{}",
            timings
                .iter()
                .map(|(name, time)| format!("{},{}", name, time))
                .collect::<Vec<String>>()
                .join("\n"),
        ),
    );
}

fn run_linear_regression() {
    let (x, y) = load_diabetes_dataset();
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);
    let mut linear_regression = LinearRegression::new();
    linear_regression.fit(&x_train, &y_train);
    let y_pred = linear_regression.predict(&x_test);
    // write y_test to ylr_test.csv and y_pred to ylr_pred.csv
    std::fs::write(
        "ylr_test.csv",
        y_test
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<String>>()
            .join("\n"),
    )
    .unwrap();
    std::fs::write(
        "ylr_pred.csv",
        y_pred
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<String>>()
            .join("\n"),
    )
    .unwrap();
}

fn run_stochastic_gd() {
    let mut sgd = StochasticGradientDescent::new(0.01, 1000);
    let (x, y) = load_diabetes_dataset();
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);
    sgd.fit(&x_train, &y_train);
    let y_pred = sgd.predict(&x_test);
    // write y_test to y_test.csv and y_pred to y_pred.csv
    std::fs::write(
        "y_test.csv",
        y_test
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<String>>()
            .join("\n"),
    )
    .unwrap();
    std::fs::write(
        "y_pred.csv",
        y_pred
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<String>>()
            .join("\n"),
    )
    .unwrap();
}

fn run_gaussian_nb() {
    // load a classification dataset
    let (x, y) = load_breast_cancer_dataset();
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);
    let classes = vec![0.0, 1.0];
    let mut gnb = GaussianNBClassifier::new(classes.clone());
    gnb.fit(&x_train, &y_train);
    let y_pred = gnb.predict(&x_test);
    let matrix = ConfusionMatrix::new(&y_test, &y_pred, &classes);
    matrix.print_matrix();
    // write matrix to matrix.csv for graphing in python
    matrix.write_to_csv("matrix.csv").unwrap();
}
fn run_gaussian_nb_iris() {
    // load a classification dataset
    let (x, y) = load_iris_dataset();
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);
    let classes = vec![0.0, 1.0, 2.0];
    let mut gnb = GaussianNBClassifier::new(classes.clone());
    gnb.fit(&x_train, &y_train);
    let y_pred = gnb.predict(&x_test);
    let matrix = ConfusionMatrix::new(&y_test, &y_pred, &classes);
    matrix.print_matrix();
    // write matrix to matrix.csv for graphing in python
    matrix.write_to_csv("matrix_iris.csv").unwrap();
}

fn run_logistic_regression() {
    let (x, y) = load_breast_cancer_dataset();
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);
    let mut logistic_regression = LogisticRegression::new(0.01, 1000);
    logistic_regression.fit(&x_train, &y_train);
    let y_pred = logistic_regression.predict(&x_test);
    let matrix = ConfusionMatrix::new(&y_test, &y_pred, &vec![0.0, 1.0]);
    matrix.print_matrix();
    matrix
        .write_to_csv("matrix_logistic_regression.csv")
        .unwrap();
}

fn run_logistic_regression_iris() {
    let (x, y) = load_iris_dataset();
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);
    let mut logistic_regression = LogisticRegression::new(0.01, 1000);
    logistic_regression.fit(&x_train, &y_train);
    let y_pred = logistic_regression.predict(&x_test);
    let matrix = ConfusionMatrix::new(&y_test, &y_pred, &vec![0.0, 1.0, 2.0]);
    matrix.print_matrix();
    matrix
        .write_to_csv("matrix_logistic_regression_iris.csv")
        .unwrap();
}

fn run_decision_tree() {
    let (x, y) = load_breast_cancer_dataset();
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);
    let mut decision_tree = DecisionTree::new(DecisionTreeClassificationMethod::InformationGain);
    decision_tree.fit(&x_train, &y_train);
    let y_pred = decision_tree.predict(&x_test);
    let matrix = ConfusionMatrix::new(&y_test, &y_pred, &vec![0.0, 1.0]);
    matrix.print_matrix();
    matrix.write_to_csv("matrix_decision_tree.csv").unwrap();
}

fn run_decision_tree_iris() {
    let (x, y) = load_iris_dataset();
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);
    let mut decision_tree = DecisionTree::new(DecisionTreeClassificationMethod::InformationGain);
    decision_tree.fit(&x_train, &y_train);
    let y_pred = decision_tree.predict(&x_test);
    let matrix = ConfusionMatrix::new(&y_test, &y_pred, &vec![0.0, 1.0, 2.0]);
    matrix.print_matrix();
    matrix
        .write_to_csv("matrix_decision_tree_iris.csv")
        .unwrap();
}

fn run_svm() {
    let (x, y) = load_breast_cancer_dataset();
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);
    // convert y from 0.0, 1.0 to -1.0, 1.0
    let y_train = y_train
        .iter()
        .map(|&yi| if yi == 0.0 { -1.0 } else { 1.0 })
        .collect();
    let mut model = SupportVectorMachine::new(1.0, 1e-4, 50, Kernel::Linear, 8);
    model.fit(x_train, y_train);
    let y_pred = model.predict(&x_test);
    let y_pred = y_pred
        .iter()
        .map(|&yi| if yi == -1.0 { 0.0 } else { 1.0 })
        .collect();
    let matrix = ConfusionMatrix::new(&y_test, &y_pred, &vec![0.0, 1.0]);
    matrix.print_matrix();
    matrix.write_to_csv("matrix_svm.csv").unwrap();
}
