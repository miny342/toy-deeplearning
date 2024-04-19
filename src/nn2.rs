use crate::matrix::Matrix;

pub trait Layer {
    fn forward(&mut self, x: Matrix) -> Matrix;
    fn backward(&mut self, x: Matrix) -> Matrix;
    fn update(&mut self, learning_rate: f64);
}

struct LeakyRelu {
    out: Matrix
}

impl LeakyRelu {
    fn new() -> Self {
        LeakyRelu { out: Matrix::zeroed((0, 0)) }
    }
}

impl Layer for LeakyRelu {
    fn forward(&mut self, x: Matrix) -> Matrix {
        self.out = x.clone().apply(|v| if v > 0.0 { 1.0 } else { 0.01 });
        x.apply(|v| if v > 0.0 { v } else { 0.01 * v })
    }
    fn backward(&mut self, x: Matrix) -> Matrix {
        x * self.out.clone()
    }
    fn update(&mut self, learning_rate: f64) {}
}

struct Affine {
    w: Matrix,
    b: Matrix,
    x: Matrix,
    dw: Matrix,
    db: Matrix,
}

impl Affine {
    fn new(w: Matrix, b: Matrix) -> Self {
        assert!(w.size().1 == b.size().1);
        assert!(b.size().0 == 1);
        Affine { w, b, x: Matrix::zeroed((0, 0)), dw: Matrix::zeroed((0, 0)), db: Matrix::zeroed((0, 0)) }
    }
    fn normal(input_size: usize, output_size: usize, mean: f64, standard: f64) -> Self {
        Affine {
            w: Matrix::normal((input_size, output_size), mean, standard),
            b: Matrix::normal((1, output_size), mean, standard),
            x: Matrix::zeroed((0, 0)),
            dw: Matrix::zeroed((0, 0)),
            db: Matrix::zeroed((0, 0))
        }
    }
}

impl Layer for Affine {
    fn forward(&mut self, x: Matrix) -> Matrix {
        self.x = x.clone();
        let mut tmp = x.dot(self.w.clone());
        for i in 0..tmp.size().0 {
            for j in 0..tmp.size().1 {
                tmp[[i, j]] += self.b[[0, j]];
            }
        }
        tmp
    }
    fn backward(&mut self, x: Matrix) -> Matrix {
        self.dw = self.x.clone().t().dot(x.clone());
        self.db = Matrix::zeroed((1, self.b.size().1));
        for i in 0..x.size().0 {
            for j in 0..x.size().1 {
                self.db[[0, j]] += x[[i, j]];
            }
        }
        x.dot(self.w.clone().t())
    }
    fn update(&mut self, learning_rate: f64) {
        self.w = self.w.clone() - self.dw.clone().apply(|v| learning_rate * v);
        self.b = self.b.clone() - self.db.clone().apply(|v| learning_rate * v);
    }
}

fn softmax(m: Matrix) -> Matrix {
    let b = m.size().0;
    let i = m.size().1;

    let mut tmp = Matrix::zeroed((b, i));
    for j in 0..b {
        let mut max = f64::MIN;
        for k in 0..i {
            if max < m[[j, k]] {
                max = m[[j, k]]
            }
        }
        let mut sum = 0.0;
        for k in 0..i {
            let t = (m[[j, k]] - max).exp();
            tmp[[j, k]] = t;
            sum += t;
        }
        for k in 0..i {
            tmp[[j, k]] /= sum;
        }
    }
    tmp
}

fn cross_entropy_error(y: Matrix, t: Matrix) -> Matrix {
    let delta = 1e-7;
    let mut tmp = Matrix::zeroed((y.size().0, 1));
    for i in 0..y.size().0 {
        for j in 0..y.size().1 {
            tmp[[i, 0]] -= t[[i, j]] * (y[[i, j]] + delta).ln();
        }
    }
    tmp
}

struct SoftmaxWithLoss {
    loss: Matrix,
    y: Matrix,
    t: Matrix,
}

impl SoftmaxWithLoss {
    fn new() -> Self {
        SoftmaxWithLoss { loss: Matrix::zeroed((0, 0)), y: Matrix::zeroed((0, 0)), t: Matrix::zeroed((0, 0)) }
    }
    fn forward(&mut self, x: Matrix, t: Matrix) -> Matrix {
        self.t = t.clone();
        self.y = softmax(x);
        self.loss = cross_entropy_error(self.y.clone(), t);
        self.loss.clone()
    }
    fn backward(&self) -> Matrix {
        self.y.clone() - self.t.clone()
    }
}

pub struct TwoLayerNet {
    layers: Vec<Box<dyn Layer>>,
    last_layer: SoftmaxWithLoss,
}

impl TwoLayerNet {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        TwoLayerNet {
            layers: vec![
                Box::new(Affine::normal(input_size, hidden_size, 0.0, 0.01)),
                Box::new(LeakyRelu::new()),
                Box::new(Affine::normal(hidden_size, output_size, 0.0, 0.01)),
            ],
            last_layer: SoftmaxWithLoss::new(),
        }
    }
    pub fn predict(&mut self, x: Matrix) -> Matrix {
        let mut v = x;
        for l in &mut self.layers {
            v = l.forward(v);
        }
        v
    }
    pub fn loss(&mut self, x: Matrix, t: Matrix) -> Matrix {
        let y = self.predict(x);
        self.last_layer.forward(y, t)
    }
    pub fn accuracy(&mut self, x: Matrix, t: Matrix) -> f64 {
        let mut cnt = 0.0;
        let y = self.predict(x);
        assert!(y.size() == t.size());
        for i in 0..y.size().0 {
            let argmax_y = {
                let mut max = 0.0;
                let mut idx = 0;
                for j in 0..y.size().1 {
                    if max < y[[i, j]] {
                        max = y[[i, j]];
                        idx = j;
                    }
                }
                idx
            };
            let argmax_t = {
                let mut max = 0.0;
                let mut idx = 0;
                for j in 0..y.size().1 {
                    if max < t[[i, j]] {
                        max = t[[i, j]];
                        idx = j;
                    }
                }
                idx
            };
            if argmax_t == argmax_y {
                cnt += 1.0;
            }
        }
        cnt / y.size().0 as f64
    }
    pub fn train(&mut self, x: Matrix, t: Matrix, learning_rate: f64) {
        self.loss(x, t);

        let mut dout = self.last_layer.backward();
        for l in self.layers.iter_mut().rev() {
            dout = l.backward(dout);
        }

        for l in self.layers.iter_mut() {
            l.update(learning_rate);
        }
    }
}
