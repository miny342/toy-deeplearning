use nn::Matrix;
mod nn;
mod matrix;

fn main() {
    let contents = std::fs::read_to_string("mnist_train.csv").unwrap();
    let contents = contents.trim().split("\n").map(|v| v.split(",").map(|v1| v1.parse::<f64>().unwrap()).collect::<Vec<f64>>()).collect::<Vec<Vec<f64>>>();

    let x_train = {
        let mut v = vec![];
        let mut m = Matrix { inner: [[0.0; 784]; 100] };
        for i in 0..600 {
            for j in 0..100 {
                for k in 0..784 {
                    m.inner[j][k] = contents[i * 100 + j][k + 1] / 255.0;
                }
            }
            v.push(m.clone());
        }
        v
    };
    let t_train = {
        let mut v = vec![];
        for i in 0..600 {
            let mut m = Matrix { inner: [[0.0; 10]; 100] };
            for j in 0..100 {
                m.inner[j][contents[i * 100 + j][0] as usize] = 1.0;
            }
            v.push(m)
        }
        v
    };
    drop(contents);

    let mut network: nn::TwoLayerNet<784, 50, 10, 100> = nn::TwoLayerNet::new();
    for l in 0..100 {
        for i in 0..600 {
            network.train(x_train[i].clone(), t_train[i].clone(), 0.01);
        }
        let acc = network.accuracy(x_train[l].clone(), t_train[l].clone());
        println!("cnt: {}", acc);
    }
}
