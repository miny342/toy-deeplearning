use std::{ops, fmt::Display};

use rand::thread_rng;
use rand_distr::{Normal, Distribution};

#[derive(Clone)]
pub struct Matrix {
    size: (usize, usize),
    inner: Vec<f64>,
}

impl Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("[").unwrap();
        for i in 0..self.size.0 {
            f.write_str("[").unwrap();
            for j in 0..self.size.1 {
                f.write_fmt(format_args!("{}", self[[i, j]])).unwrap();
                if j != self.size.1 - 1 {
                    f.write_str(", ").unwrap()
                } else {
                    f.write_str("]").unwrap()
                }
            }
            if i != self.size.0 - 1 {
                f.write_str("\n").unwrap();
            }
        }
        f.write_str("]")
    }
}

impl ops::Index<[usize; 2]> for Matrix {
    type Output = f64;
    fn index(&self, index: [usize; 2]) -> &Self::Output {
        &self.inner[index[0] * self.size.1 + index[1]]
    }
}

impl ops::IndexMut<[usize; 2]> for Matrix {
    fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
        &mut self.inner[index[0] * self.size.1 + index[1]]
    }
}

impl ops::Add for Matrix {
    type Output = Matrix;
    fn add(self, rhs: Self) -> Self::Output {
        self.size_check(&rhs);
        Matrix::new_onedim_from_iter(
            self.size,
            self.inner.iter().zip(rhs.inner.iter()).map(|(&x, &y)| x + y)
        )
    }
}

impl ops::Sub for Matrix {
    type Output = Matrix;
    fn sub(self, rhs: Self) -> Self::Output {
        self.size_check(&rhs);
        Matrix::new_onedim_from_iter(
            self.size,
            self.inner.iter().zip(rhs.inner.iter()).map(|(&x, &y)| x - y)
        )
    }
}

impl ops::Mul for Matrix {
    type Output = Matrix;
    fn mul(self, rhs: Self) -> Self::Output {
        self.size_check(&rhs);
        Matrix::new_onedim_from_iter(
            self.size,
            self.inner.iter().zip(rhs.inner.iter()).map(|(&x, &y)| x * y)
        )
    }
}

impl Matrix {
    fn size_check(&self, rhs: &Self) {
        if self.size != rhs.size {
            panic!("size: {:?} != {:?}", self.size, rhs.size);
        }
    }
    pub fn normal(size: (usize, usize), mean: f64, standard: f64) -> Matrix {
        let mut rng = thread_rng();
        let normal = Normal::<f64>::new(mean, standard).unwrap();
        Matrix::new_onedim_from_iter(size, (0..size.0*size.1).map(|_| normal.sample(&mut rng)))
    }
    pub fn zeroed(size: (usize, usize)) -> Matrix {
        Matrix { size, inner: vec![0.0; size.0 * size.1] }
    }
    pub fn new_onedim(size: (usize, usize), it: &[f64]) -> Matrix {
        let v = it.to_vec();
        assert!(v.len() == size.0 * size.1);
        Matrix { size, inner: v }
    }
    fn new_onedim_from_iter(size: (usize, usize), it: impl Iterator<Item=f64>) -> Matrix {
        let mut v = Vec::with_capacity(size.0 * size.1);
        it.for_each(|x| v.push(x));
        Matrix { size, inner: v }
    }
    pub fn dot(self, rhs: Self) -> Matrix {
        if self.size.1 != rhs.size.0 {
            panic!("lhs col != rhs row, size: {:?}, {:?}", self.size, rhs.size);
        } else {
            let mut v = Matrix::zeroed((self.size.0, rhs.size.1));
            for i in 0..self.size.0 {
                for j in 0..rhs.size.1 {
                    for k in 0..self.size.1 {
                        v[[i, j]] += self[[i, k]] * rhs[[k, j]]
                    }
                }
            }
            v
        }
    }
    pub fn t(self) -> Matrix {
        let mut m = Matrix::zeroed((self.size.1, self.size.0));
        for i in 0..self.size.0 {
            for j in 0..self.size.1 {
                m[[j, i]] = self[[i, j]];
            }
        }
        m
    }
    pub fn apply(mut self, mut f: impl FnMut(f64) -> f64) -> Matrix {
        self.inner.iter_mut().for_each(|v| *v = f(*v));
        self
    }
    pub fn size(&self) -> (usize, usize) {
        self.size
    }
}


