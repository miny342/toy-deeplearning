mod nn;
mod matrix;
mod nn2;

const B: usize = 100;

fn main() {
    let contents = std::fs::read_to_string("mnist_train.csv").unwrap();
    let contents = contents.trim().split("\n").map(|v| v.split(",").map(|v1| v1.parse::<f64>().unwrap()).collect::<Vec<f64>>()).collect::<Vec<Vec<f64>>>();

    let tests = std::fs::read_to_string("mnist_test.csv").unwrap();
    let tests = tests.trim().split("\n").map(|v| v.split(",").map(|v1| v1.parse::<f64>().unwrap()).collect::<Vec<f64>>()).collect::<Vec<Vec<f64>>>();

    let x_train = {
        let mut v = vec![];
        let mut m = matrix::Matrix::zeroed((B, 784));
        for i in 0..60000 / B {
            for j in 0..B {
                for k in 0..784 {
                    m[[j, k]] = contents[i * B + j][k + 1] / 255.0;
                }
            }
            v.push(m.clone());
        }
        v
    };
    let t_train = {
        let mut v = vec![];
        for i in 0..60000 / B {
            let mut m = matrix::Matrix::zeroed((B, 10));
            for j in 0..B {
                m[[j, contents[i * B + j][0] as usize]] = 1.0;
            }
            v.push(m)
        }
        v
    };
    drop(contents);

    let x_test = {
        let mut v = vec![];
        let mut m = matrix::Matrix::zeroed((B, 784));
        for i in 0..10000 / B {
            for j in 0..B {
                for k in 0..784 {
                    m[[j, k]] = tests[i * B + j][k + 1] / 255.0;
                }
            }
            v.push(m.clone());
        }
        v
    };
    let t_test = {
        let mut v = vec![];
        for i in 0..10000 / B {
            let mut m = matrix::Matrix::zeroed((B, 10));
            for j in 0..B {
                m[[j, tests[i * B + j][0] as usize]] = 1.0;
            }
            v.push(m)
        }
        v
    };
    drop(tests);

    let mut network: nn2::TwoLayerNet = nn2::TwoLayerNet::new(784, 50, 10);
    for l in 0..1 {
        for i in 0..60000 / B {
            network.train(x_train[i].clone(), t_train[i].clone(), 0.01);
            // println!("loss: {}", network.loss(x_train[i].clone(), t_train[i].clone()));
        }
        let mut acc = 0.0;
        for i in 0..60000 / B {
            acc += network.accuracy(x_train[i].clone(), t_train[i].clone())
        }
        let mut tacc = 0.0;
        for i in 0..10000 / B {
            tacc += network.accuracy(x_test[i].clone(), t_test[i].clone())
        }
        println!("accuracy: {} {}", acc / 60000.0 * B as f64, tacc / 10000.0 * B as f64);
    }
}
