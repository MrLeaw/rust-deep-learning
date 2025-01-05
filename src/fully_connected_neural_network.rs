use rand::Rng;
use std::fmt;

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn relu(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

fn linear(x: f64) -> f64 {
    x
}

fn leaky_relu(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        0.01 * x
    }
}

fn softmax(x: Vec<f64>) -> Vec<f64> {
    let sum: f64 = x.iter().map(|x| x.exp()).sum();
    x.iter().map(|x| x.exp() / sum).collect()
}

#[derive(Clone, Debug)]
enum Activation {
    Sigmoid,
    Relu,
    Linear,
    LeakyRelu,
    Softmax,
}

struct ModelBuilder {
    num_input_nodes: usize,
    hidden_layers: Vec<(usize, Activation)>,
    output_nodes: (usize, Activation),
}

impl ModelBuilder {
    pub fn new() -> ModelBuilder {
        ModelBuilder {
            num_input_nodes: 0,
            hidden_layers: Vec::new(),
            output_nodes: (0, Activation::Sigmoid),
        }
    }

    pub fn input_nodes(&mut self, num_input_nodes: usize) -> &mut Self {
        self.num_input_nodes = num_input_nodes;
        self
    }

    pub fn hidden_layer(&mut self, num_nodes: usize, activation: Activation) -> &mut Self {
        self.hidden_layers.push((num_nodes, activation));
        self
    }

    pub fn output_nodes(&mut self, num_output_nodes: usize, activation: Activation) -> &mut Self {
        self.output_nodes = (num_output_nodes, activation);
        self
    }

    pub fn build(&self) -> NeuralNetwork {
        let nn = NeuralNetwork::new(
            self.num_input_nodes,
            self.hidden_layers.clone(),
            self.output_nodes.clone(),
        );
        nn
    }
}

impl Activation {
    fn activate(&self, x: f64) -> f64 {
        match self {
            Activation::Sigmoid => sigmoid(x),
            Activation::Relu => relu(x),
            Activation::Linear => linear(x),
            Activation::LeakyRelu => leaky_relu(x),
            Activation::Softmax => panic!("Softmax not implemented for single values"),
        }
    }

    fn activate_vector(&self, x: Vec<f64>) -> Vec<f64> {
        match self {
            Activation::Softmax => softmax(x),
            _ => x.iter().map(|x| self.activate(*x)).collect(),
        }
    }

    fn derivative(&self, x: f64) -> f64 {
        match self {
            Activation::Sigmoid => sigmoid(x) * (1.0 - sigmoid(x)),
            Activation::Relu => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Activation::Linear => 1.0,
            Activation::LeakyRelu => {
                if x > 0.0 {
                    1.0
                } else {
                    0.01
                }
            }
            Activation::Softmax => panic!("Softmax not implemented for single values"),
        }
    }
}

#[derive(Clone)]
struct Node {
    weights: Vec<f64>,
    bias: f64,
}

impl fmt::Debug for Node {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Node {{ weights count: {}, bias: {} }}\n",
            self.weights.len(),
            self.bias
        )
    }
}

impl Node {
    fn new(num_weights: usize) -> Node {
        let mut rng = rand::thread_rng();
        let weights: Vec<f64> = (0..num_weights).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let bias = rng.gen_range(-1.0..1.0);
        Node { weights, bias }
    }

    fn forward(&self, inputs: &Vec<f64>, activation: &Activation) -> f64 {
        // Check that the number of weights matches the number of inputs
        if inputs.len() != self.weights.len() {
            panic!(
                "Input size ({}) does not match weights size ({}) in Node",
                inputs.len(),
                self.weights.len()
            );
        }

        // Compute weighted sum
        let sum: f64 = inputs
            .iter()
            .zip(self.weights.iter())
            .map(|(input, weight)| input * weight)
            .sum::<f64>()
            + self.bias;

        // Return sum directly as some activation functions are applied to the whole vector
        sum
    }
}
struct NeuralNetwork {
    num_input_nodes: usize,
    hidden_layers: Vec<(Vec<Node>, Activation)>,
    output_nodes: (Vec<Node>, Activation),
}
impl fmt::Debug for NeuralNetwork {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "NeuralNetwork {{ input_nodes count: {:?},\nhidden_layers: {:?},\noutput_nodes: {:?} }}",
           self.num_input_nodes, self.hidden_layers, self.output_nodes
        )
    }
}
impl NeuralNetwork {
    fn new(
        num_input_nodes: usize,
        hidden_layer_layout: Vec<(usize, Activation)>,
        output_node_count_activation: (usize, Activation),
    ) -> NeuralNetwork {
        // Initialize hidden layers
        let mut hidden_layers: Vec<(Vec<Node>, Activation)> = Vec::new();
        let mut prev_layer_size = num_input_nodes; // Start with the input size
        for (num_nodes, activation) in hidden_layer_layout {
            let nodes: Vec<Node> = (0..num_nodes)
                .map(|_| Node::new(prev_layer_size)) // Use prev_layer_size for weights
                .collect();
            hidden_layers.push((nodes, activation));
            prev_layer_size = num_nodes; // Update for the next layer
        }

        // Initialize output node(s)
        let output_nodes = (0..output_node_count_activation.0)
            .map(|_| Node::new(prev_layer_size)) // Use prev_layer_size for weights
            .collect();

        NeuralNetwork {
            num_input_nodes,
            hidden_layers,
            output_nodes: (output_nodes, output_node_count_activation.1),
        }
    }

    pub fn predict(&self, inputs: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        inputs.iter().map(|x| self.predict_one(x.clone())).collect()
    }

    pub fn predict_one(&self, inputs: Vec<f64>) -> Vec<f64> {
        self.forward(&inputs)
    }

    // forward pass through the network and return the output
    fn forward(&self, inputs: &Vec<f64>) -> Vec<f64> {
        let mut layer_inputs = inputs.clone(); // Start with the input to the network
        for (nodes, activation) in &self.hidden_layers {
            // print input count, weights count and output count
            let mut outputs = Vec::new();
            for node in nodes {
                outputs.push(node.forward(&layer_inputs, activation)); // Use outputs of the previous layer
            }
            layer_inputs = activation.activate_vector(outputs); // Outputs of the current layer
                                                                // become inputs for the next
        }
        let mut outputs: Vec<f64> = Vec::new();
        for node in &self.output_nodes.0 {
            outputs.push(node.forward(&layer_inputs, &self.output_nodes.1));
        }
        self.output_nodes.1.activate_vector(outputs)
    }

    fn train(&mut self, train_x: Vec<Vec<f64>>, train_y: Vec<Vec<f64>>, lr: f64, iters: u64) {
        // check that all targets have the same length and also that the length is the same as the
        // number of output nodes
        if !train_y.iter().all(|x| x.len() == train_y[0].len())
            || train_y[0].len() != self.output_nodes.0.len()
        {
            panic!("Targets have different lengths or the length is not the same as the number of output nodes");
        }
        let ind = indicatif::ProgressBar::new(iters);
        for _ in 0..iters {
            for (inputs, target) in train_x.iter().zip(train_y.iter()) {
                self.train_iter(inputs, target, lr);
            }
            ind.inc(1);
        }
        ind.finish();
    }

    fn mse(&self, x_train: Vec<Vec<f64>>, y_train: Vec<f64>) -> f64 {
        let results: Vec<Vec<f64>> = x_train.iter().map(|x| self.forward(x)).collect();
        let diffs: Vec<Vec<f64>> = results
            .iter()
            .zip(y_train.iter())
            .map(|(result, target)| vec![result[0] - target])
            .collect();

        // print metrics (MSE)
        let mse = diffs.iter().map(|x| x[0].powi(2)).sum::<f64>() / diffs.len() as f64;
        mse
    }

    fn train_iter(&mut self, inputs: &Vec<f64>, targets: &Vec<f64>, lr: f64) {
        // Forward pass: compute outputs layer by layer
        let mut layer_inputs = inputs.clone();
        let mut layer_outputs: Vec<Vec<f64>> = Vec::new(); // Store outputs of each layer
        for (nodes, activation) in &self.hidden_layers {
            let mut outputs = Vec::new();
            for node in nodes {
                outputs.push(node.forward(&layer_inputs, activation));
            }
            layer_outputs.push(outputs.clone());
            layer_inputs = outputs; // Outputs of the current layer become inputs for the next
        }

        // Forward pass for output layer
        let mut output_layer_inputs = layer_outputs.last().unwrap_or(&inputs).clone();
        let outputs: Vec<f64> = self
            .output_nodes
            .0
            .iter()
            .map(|node| node.forward(&output_layer_inputs, &self.output_nodes.1))
            .collect();

        // Compute output error (difference between predictions and targets)
        let output_errors: Vec<f64> = targets
            .iter()
            .zip(outputs.iter())
            .map(|(target, output)| target - output)
            .collect();

        // Compute error signals for the output layer
        let output_activation = &self.hidden_layers.last().unwrap().1;
        let output_error_signals: Vec<f64> = outputs
            .iter()
            .zip(output_errors.iter())
            .map(|(output, error)| error * output_activation.derivative(*output))
            .collect();

        // Update weights and biases for the output layer
        for (node, error_signal) in self
            .output_nodes
            .0
            .iter_mut()
            .zip(output_error_signals.iter())
        {
            for (i, weight) in node.weights.iter_mut().enumerate() {
                *weight += lr * error_signal * output_layer_inputs[i];
            }
            node.bias += lr * error_signal;
        }

        // Backpropagate errors through the hidden layers
        let mut next_layer_error_signals = output_error_signals.clone(); // Start with output layer signals
        for layer_idx in (0..self.hidden_layers.len()).rev() {
            let activation = &self.hidden_layers[layer_idx].1;
            let mut current_layer_error_signals = vec![0.0; self.hidden_layers[layer_idx].0.len()];
            let previous_layer_outputs = if layer_idx == 0 {
                inputs
            } else {
                &layer_outputs[layer_idx - 1]
            };

            // Compute error signals for the current layer
            for (i, _) in self.hidden_layers[layer_idx].0.iter().enumerate() {
                let mut error_sum = 0.0;
                if layer_idx == self.hidden_layers.len() - 1 {
                    // Connected to output layer
                    for (j, output_node) in self.output_nodes.0.iter().enumerate() {
                        error_sum += output_node.weights[i] * next_layer_error_signals[j];
                    }
                } else {
                    // Connected to the next hidden layer
                    let (next_nodes, _) = &self.hidden_layers[layer_idx + 1];
                    for (j, next_node) in next_nodes.iter().enumerate() {
                        error_sum += next_node.weights[i] * next_layer_error_signals[j];
                    }
                }
                current_layer_error_signals[i] =
                    error_sum * activation.derivative(layer_outputs[layer_idx][i]);
            }

            // Update weights and biases for the current layer
            for (i, node) in self.hidden_layers[layer_idx].0.iter_mut().enumerate() {
                for j in 0..node.weights.len() {
                    node.weights[j] +=
                        lr * current_layer_error_signals[i] * previous_layer_outputs[j];
                }
                node.bias += lr * current_layer_error_signals[i];
            }

            // Update error signals for the next layer (moving backward)
            next_layer_error_signals = current_layer_error_signals;
        }
    }
}

fn normalize_x_y(x: &mut Vec<Vec<f64>>, y: &mut Vec<f64>) {
    normalize_column(y);
    // each item of x is a ROW. We need to normalize each column
    let mut x_normalized: Vec<Vec<f64>> = Vec::new();
    for column in transpose(&x) {
        let mut column = column.clone();
        normalize_column(&mut column);
        x_normalized.push(column);
    }
    *x = transpose(&x_normalized);
}

fn normalize_column(column: &mut Vec<f64>) {
    let min = column.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = column.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    for i in 0..column.len() {
        column[i] = (column[i] - min) / (max - min);
    }
}

fn transpose(x: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut res = vec![vec![0.0; x.len()]; x[0].len()];
    for i in 0..x.len() {
        for j in 0..x[0].len() {
            res[j][i] = x[i][j];
        }
    }
    res
}

#[cfg(test)]
mod tests {
    use crate::datasets::{load_boston_dataset, train_test_split};

    use super::*;

    #[test]
    fn test_sigmoid() {
        assert_eq!(sigmoid(0.0), 0.5);
    }

    #[test]
    fn test_relu() {
        assert_eq!(relu(0.0), 0.0);
        assert_eq!(relu(1.0), 1.0);
        assert_eq!(relu(-1.0), 0.0);
    }

    #[test]
    fn test_activation() {
        let activation = Activation::Sigmoid;
        assert_eq!(activation.activate(0.0), 0.5);
        let activation = Activation::Relu;
        assert_eq!(activation.activate(0.0), 0.0);
    }

    #[test]
    fn test_node_forward() {
        let node = Node::new(3);
        let inputs = vec![1.0, 2.0, 3.0];
        let activation = Activation::Sigmoid;
        let output = node.forward(&inputs, &activation);
        assert!(output >= 0.0 && output <= 1.0);
    }

    #[test]
    fn test_neural_network_forward() {
        let nn = ModelBuilder::new()
            .input_nodes(3)
            .hidden_layer(3, Activation::Sigmoid)
            .output_nodes(1, Activation::Sigmoid)
            .build();
        let inputs = vec![1.0, 2.0, 3.0];
        let output = nn.predict_one(inputs)[0];
        assert!(output >= 0.0 && output <= 1.0);
    }

    #[test]
    fn test_neural_network_train() {
        let mut nn = ModelBuilder::new()
            .input_nodes(10)
            .hidden_layer(20, Activation::Sigmoid)
            .output_nodes(1, Activation::Sigmoid)
            .build();
        let train_x = vec![vec![0.0; 10], vec![1.0; 10]];
        let train_y = vec![vec![0.0], vec![1.0]];
        nn.train(train_x, train_y, 0.1, 1000);
        println!("{:?}", nn.predict_one(vec![0.0; 10]));
        assert!(nn.predict_one(vec![0.0; 10])[0] < 0.3);
    }

    #[test]
    fn test_ms_error() {
        let (mut x, mut y) = load_boston_dataset();
        normalize_x_y(&mut x, &mut y);
        let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);
        let y_train = y_train.iter().map(|&x| vec![x]).collect();
        let y_test = y_test.iter().map(|&x| x).collect();
        let mut nn = ModelBuilder::new()
            .input_nodes(13)
            .hidden_layer(20, Activation::Sigmoid)
            .output_nodes(1, Activation::Sigmoid)
            .build();
        nn.train(x_train, y_train, 0.1, 1000);
        let mse = nn.mse(x_test, y_test);
        println!("{}", mse);
        assert!(mse < 0.08);
    }

    #[test]
    #[should_panic]
    fn test_mismatched_targets() {
        let mut nn = ModelBuilder::new()
            .input_nodes(10)
            .hidden_layer(20, Activation::Sigmoid)
            .output_nodes(1, Activation::Sigmoid)
            .build();
        let train_x = vec![vec![0.0; 10], vec![1.0; 10]];
        let train_y = vec![vec![0.0], vec![1.0, 1.0]];
        nn.train(train_x, train_y, 0.1, 1000);
    }

    #[test]
    #[should_panic]
    fn test_mismatched_output_nodes() {
        let mut nn = ModelBuilder::new()
            .input_nodes(10)
            .hidden_layer(20, Activation::Sigmoid)
            .output_nodes(2, Activation::Sigmoid)
            .build();
        let train_x = vec![vec![0.0; 10], vec![1.0; 10]];
        let train_y = vec![vec![0.0], vec![1.0]];
        nn.train(train_x, train_y, 0.1, 1000);
    }

    #[test]
    fn multiple_hidden_layers() {
        let mut nn = ModelBuilder::new()
            .input_nodes(13)
            .hidden_layer(20, Activation::Relu)
            .output_nodes(1, Activation::Sigmoid)
            .build();
        let (mut x, mut y) = load_boston_dataset();
        normalize_x_y(&mut x, &mut y);
        let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);
        nn.train(
            x_train,
            y_train.iter().map(|&x| vec![x]).collect(),
            0.1,
            500,
        );
        let output = nn.predict(x_test);
        for (output, target) in output.iter().zip(y_test.iter()) {
            println!("{:?} target {:?}", output, target);
        }
        let mse = output
            .iter()
            .zip(y_test.iter())
            .map(|(output, target)| (output[0] - target).powi(2))
            .sum::<f64>()
            / y_test.len() as f64;
        println!("{}", mse);
    }
}
