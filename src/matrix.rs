use std::ops;



struct Matrix {
    size: (usize, usize),
    inner: Vec<f64>,
}

impl ops::Index<usize> for Matrix {
    type Output = [f64];
    fn index(&self, index: usize) -> &Self::Output {
        &self.inner[index * self.size.1..(index + 1) * self.size.1]
    }
}

impl ops::IndexMut<usize> for Matrix {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.inner[index * self.size.1..(index + 1) * self.size.1]
    }
}

impl Matrix {
    fn zeroed(size: (usize, usize)) -> Matrix {
        Matrix { size, inner: vec![0.0; size.0 * size.1] }
    }
    fn new_onedim(size: (usize, usize), it: impl Iterator<Item = f64>) -> Matrix {
        let mut v = Vec::with_capacity(size.0 * size.1);
        for i in it {
            v.push(i);
        }
        assert!(v.len() == size.0 * size.1);
        Matrix { size, inner: v }
    }
    fn new(size: (usize, usize), it: impl Iterator<Item = impl Iterator<Item = f64>>) -> Matrix {
        Self::new_onedim(size, it.flatten())
    }
    fn dot(&self, rhs: &Self) -> Option<Matrix> {
        if self.size.1 != rhs.size.0 {
            None
        } else {
            let mut v = Matrix::zeroed((self.size.0, rhs.size.1));
            for i in 0..self.size.0 {
                for j in 0..rhs.size.1 {
                    for k in 0..self.size.1 {
                        v[i][j] += self[i][k] * rhs[k][j]
                    }
                }
            }
            Some(v)
        }
    }
    fn add_(&self, rhs: &Self) -> Option<Matrix> {
        if self.size != rhs.size {
            None
        } else {
            Some(Self::new_onedim(
                self.size,
                self.inner.iter().zip(rhs.inner.iter()).map(|(&v1, &v2)| v1 + v2)
            ))
        }
    }
    fn sub_(&self, rhs: &Self) -> Option<Matrix> {
        if self.size != rhs.size {
            None
        } else {
            Some(Self::new_onedim(
                self.size,
                self.inner.iter().zip(rhs.inner.iter()).map(|(&v1, &v2)| v1 - v2)
            ))
        }
    }
    fn mul_(&self, rhs: &Self) -> Option<Matrix> {
        if self.size != rhs.size {
            None
        } else {
            Some(Self::new_onedim(
                self.size,
                self.inner.iter().zip(rhs.inner.iter()).map(|(&v1, &v2)| v1 * v2)
            ))
        }
    }
}


