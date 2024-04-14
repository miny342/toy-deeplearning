use std::{ops::{Add, Sub, Mul}, fmt::Display};

use rand::thread_rng;
use rand_distr::{Normal, Distribution};

#[derive(Clone)]
pub struct Matrix<const R: usize, const C: usize> {
    pub inner: [[f64; C]; R]
}

impl<const R: usize, const C: usize> Display for Matrix<R, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("[").unwrap();
        for i in 0..R {
            f.write_fmt(format_args!("{:?}\n", self.inner[i])).unwrap();
        }
        f.write_str("]")
    }
}

impl<const R: usize, const C: usize> Matrix<R, C> {
    fn zeroed() -> Self {
        Matrix { inner: [[0.0; C]; R] }
    }
    fn normal() -> Self {
        let mut rng = thread_rng();
        let normal = Normal::<f64>::new(0.0, 0.01).unwrap();
        Matrix { inner: std::array::from_fn(|_| std::array::from_fn(|_| normal.sample(&mut rng))) }
    }
    fn attach<F: FnMut(f64) -> f64>(&self, mut f: F) -> Matrix<R, C> {
        let mut m = [[0.0; C]; R];
        for i in 0..R {
            for j in 0..C {
                m[i][j] = f(self.inner[i][j]);
            }
        }
        Matrix { inner: m }
    }
    pub fn dot<const Z: usize>(self, rhs: Matrix<C, Z>) -> Matrix<R, Z> {
        let mut m = [[0.0; Z]; R];
        for i in 0..R {
            for j in 0..Z {
                for k in 0..C {
                    m[i][j] += self.inner[i][k] * rhs.inner[k][j];
                }
            }
        }
        Matrix { inner: m }
    }
    pub fn t(self) -> Matrix<C, R> {
        let mut m = [[0.0; R]; C];
        for i in 0..R {
            for j in 0..C {
                m[j][i] = self.inner[i][j]
            }
        }
        Matrix { inner: m }
    }
    fn sum(&self) -> f64 {
        let mut v = 0.0;
        for i in 0..R {
            for j in 0..C {
                v += self.inner[i][j];
            }
        }
        v
    }
}


impl<const X: usize, const Y: usize> Add for Matrix<X, Y> {
    type Output = Matrix<X, Y>;
    fn add(self, rhs: Self) -> Self::Output {
        let mut m = [[0.0; Y]; X];
        for i in 0..X {
            for j in 0..Y {
                m[i][j] = self.inner[i][j] + rhs.inner[i][j];
            }
        }
        Matrix { inner: m }
    }
}

impl<const X: usize, const Y: usize> Sub for Matrix<X, Y> {
    type Output = Matrix<X, Y>;
    fn sub(self, rhs: Self) -> Self::Output {
        let mut m = [[0.0; Y]; X];
        for i in 0..X {
            for j in 0..Y {
                m[i][j] = self.inner[i][j] - rhs.inner[i][j];
            }
        }
        Matrix { inner: m }
    }
}

impl<const R: usize, const C: usize> Sub<Matrix<R, C>> for f64 {
    type Output = Matrix<R, C>;
    fn sub(self, rhs: Matrix<R, C>) -> Self::Output {
        let mut m = [[0.0; C]; R];
        for i in 0..R {
            for j in 0..C {
                m[i][j] = self - rhs.inner[i][j];
            }
        }
        Matrix { inner: m }
    }
}

impl<const X: usize, const Y: usize> Mul for Matrix<X, Y> {
    type Output = Matrix<X, Y>;
    fn mul(self, rhs: Matrix<X, Y>) -> Self::Output {
        let mut m = [[0.0; Y]; X];
        for i in 0..X {
            for j in 0..Y {
                m[i][j] = self.inner[i][j] * rhs.inner[i][j];
            }
        }
        Matrix { inner: m }
    }
}

impl<const X: usize, const Y: usize> Mul<Matrix<X, Y>> for f64 {
    type Output = Matrix<X, Y>;
    fn mul(self, rhs: Matrix<X, Y>) -> Self::Output {
        let mut m = [[0.0; Y]; X];
        for i in 0..X {
            for j in 0..Y {
                m[i][j] = self * rhs.inner[i][j];
            }
        }
        Matrix { inner: m }
    }
}

trait Layer<const R: usize, const C: usize, const OR: usize = R, const OC: usize = C> {
    fn forward(&mut self, x: Matrix<R, C>) -> Matrix<OR, OC>;
    fn backward(&mut self, dout: Matrix<OR, OC>) -> Matrix<R, C>;
}

struct LeakyRelu<const R: usize, const C: usize> {
    out: Matrix<R, C>
}

impl<const R: usize, const C: usize> LeakyRelu<R, C> {
    fn new() -> Self {
        LeakyRelu { out: Matrix::zeroed() }
    }
}

impl<const R: usize, const C: usize> Layer<R, C> for LeakyRelu<R, C> {
    fn forward(&mut self, x: Matrix<R, C>) -> Matrix<R, C> {
        self.out = x.attach(|v| if v > 0.0 { 1.0 } else { 0.01 });
        x.attach(|v| if v > 0.0 { v } else { 0.01 * v })
    }
    fn backward(&mut self, dout: Matrix<R, C>) -> Matrix<R, C> {
        dout * self.out.clone()
    }
}

struct Sigmoid<const R: usize, const C: usize> {
    out: Matrix<R, C>
}

impl<const R: usize, const C: usize> Sigmoid<R, C> {
    fn new() -> Self {
        Sigmoid { out: Matrix::zeroed() }
    }
}

impl<const R: usize, const C: usize> Layer<R, C> for Sigmoid<R, C> {
    fn forward(&mut self, x: Matrix<R, C>) -> Matrix<R, C> {
        self.out = x.attach(|v| 1.0 / (1.0 + (-v).exp()));
        self.out.clone()
    }
    fn backward(&mut self, dout: Matrix<R, C>) -> Matrix<R, C> {
        dout * (1.0 - self.out.clone()) * self.out.clone()
    }
}

struct Affine<const I: usize, const O: usize, const B: usize> {
    w: Matrix<I, O>,
    b: Matrix<1, O>,
    x: Matrix<B, I>,
    dw: Matrix<I, O>,
    db: Matrix<1, O>,
}

impl<const I: usize, const O: usize, const B: usize> Affine<I, O, B> {
    fn new(w: Matrix<I, O>, b: Matrix<1, O>) -> Self {
        Affine { w, b, x: Matrix::zeroed(), dw: Matrix::zeroed(), db: Matrix::zeroed() }
    }
}

impl<const I: usize, const O: usize, const B: usize> Layer<B, I, B, O> for Affine<I, O, B> {
    fn forward(&mut self, x: Matrix<B, I>) -> Matrix<B, O> {
        self.x = x.clone();
        let mut tmp = x.dot(self.w.clone());
        for i in 0..B {
            for j in 0..O {
                tmp.inner[i][j] += self.b.inner[0][j];
            }
        }
        tmp
    }
    fn backward(&mut self, dout: Matrix<B, O>) -> Matrix<B, I> {
        self.dw = self.x.clone().t().dot(dout.clone());
        self.db = Matrix::zeroed();
        for i in 0..B {
            for j in 0..O {
                self.db.inner[0][j] += dout.inner[i][j];
            }
        }
        dout.dot(self.w.clone().t())
    }
}

fn softmax<const I: usize, const B: usize>(m: Matrix<B, I>) -> Matrix<B, I> {
    // let mut max = f64::MIN;
    // for i in 0..I {
    //     if max < m.inner[0][i] {
    //         max = m.inner[0][i]
    //     }
    // }
    // let m = m.attach(|v| (v - max).exp());
    // m.attach(|v| v / m.sum())
    let mut tmp = Matrix::<B, I>::zeroed();
    for i in 0..B {
        let mut max = f64::MIN;
        for j in 0..I {
            if max < m.inner[i][j] {
                max = m.inner[i][j];
            }
        }
        let mut sum = 0.0;
        for j in 0..I {
            let t = (m.inner[i][j] - max).exp();
            tmp.inner[i][j] = t;
            sum += t;
        }
        for j in 0..I {
            tmp.inner[i][j] /= sum;
        }
    }
    tmp
}

fn cross_entropy_error<const I: usize, const B: usize>(y: Matrix<B, I>, t: Matrix<B, I>) -> Matrix<B, 1> {
    // let delta = 1e-7;
    // -(t * y.attach(|v| (v + delta).ln())).sum()
    let delta = 1e-7;
    let mut tmp = Matrix::<B, 1>::zeroed();
    for i in 0..B {
        for j in 0..I {
            tmp.inner[i][0] -= t.inner[i][j] * (y.inner[i][j] + delta).ln();
        }
    }
    tmp
}

struct SoftmaxWithLoss<const I: usize, const B: usize> {
    loss: Matrix<B, 1>,
    y: Matrix<B, I>,
    t: Matrix<B, I>,
}

impl<const I: usize, const B: usize> SoftmaxWithLoss<I, B> {
    fn new() -> Self {
        SoftmaxWithLoss { loss: Matrix::zeroed(), y: Matrix::zeroed(), t: Matrix::zeroed() }
    }
    fn forward(&mut self, x: Matrix<B, I>, t: Matrix<B, I>) -> Matrix<B, 1> {
        self.t = t.clone();
        self.y = softmax(x);
        self.loss = cross_entropy_error(self.y.clone(), t);
        self.loss.clone()
    }
    fn backward(&self) -> Matrix<B, I> {
        self.y.clone() - self.t.clone()
    }
}

pub struct TwoLayerNet<const I: usize, const H: usize, const O: usize, const B: usize> {
    affine1: Affine<I, H, B>,
    relu: LeakyRelu<B, H>,
    affine2: Affine<H, O, B>,
    last: SoftmaxWithLoss<O, B>,
}

impl<const I: usize, const H: usize, const O: usize, const B: usize> TwoLayerNet<I, H, O, B> {
    pub fn new() -> Self {
        TwoLayerNet {
            affine1: Affine::new(Matrix::normal(), Matrix::normal()),
            relu: LeakyRelu::new(),
            affine2: Affine::new(Matrix::normal(), Matrix::normal()),
            last: SoftmaxWithLoss::new(),
        }
    }
    pub fn predict(&mut self, x: Matrix<B, I>) -> Matrix<B, O> {
        let v = self.affine1.forward(x);
        let v = self.relu.forward(v);
        let v = self.affine2.forward(v);
        v
    }
    pub fn loss(&mut self, x: Matrix<B, I>, t: Matrix<B, O>) -> Matrix<B, 1> {
        let y = self.predict(x);
        self.last.forward(y, t)
    }
    pub fn accuracy(&mut self, x: Matrix<B, I>, t: Matrix<B, O>) -> f64 {
        let mut cnt = 0.0;
        let y = self.predict(x);
        for j in 0..B {
            let argmaxy = {
                let mut max = 0.0;
                let mut idx = 0;
                for i in 0..O {
                    if max < y.inner[j][i] {
                        max = y.inner[j][i];
                        idx = i;
                    }
                }
                idx
            };
            let argmaxt = {
                let mut max = 0.0;
                let mut idx = 0;
                for i in 0..O {
                    if max < t.inner[j][i] {
                        max = t.inner[j][i];
                        idx = i;
                    }
                }
                idx
            };
            if argmaxy == argmaxt {
                cnt += 1.0;
            }
        }
        cnt / (B as f64)
    }
    pub fn train(&mut self, x: Matrix<B, I>, t: Matrix<B, O>, learning_rate: f64) {
        // forward
        self.loss(x, t);

        // backward
        let dout = self.last.backward();

        let dout = self.affine2.backward(dout);
        let dout = self.relu.backward(dout);
        self.affine1.backward(dout);

        // update
        self.affine1.w = self.affine1.w.clone() - learning_rate * self.affine1.dw.clone();
        self.affine1.b = self.affine1.b.clone() - learning_rate * self.affine1.db.clone();

        self.affine2.w = self.affine2.w.clone() - learning_rate * self.affine2.dw.clone();
        self.affine2.b = self.affine2.b.clone() - learning_rate * self.affine2.db.clone();
    }
}
