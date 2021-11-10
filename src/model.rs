use std::{marker::PhantomData, ops::{AddAssign, DivAssign, MulAssign, SubAssign}};

pub use na::{SMatrix, SVector};

pub struct Model<L, F, C, const I: usize, const O: usize>
where
    L: Layers<F, C, I, O>,
    C: Calculation<O>
{
    pub layers: L,
    _maker: PhantomData<(F,C)>,
}

impl<L, F, C, const I: usize, const O: usize> Model<L, F, C, I, O> 
where
    L: Layers<F, C, I, O>,
    C: Calculation<O>
{
    #[inline]
    pub fn new(layers: L) -> Self {
        Self {
            layers,
            _maker: Default::default()
        }
    }

    #[inline]
    pub fn test(&self, item: &SVector<f64, I>) -> SVector<f64,O> {
        self.layers.test(item)
    }
}

/// `I`: input vec size
/// 
/// `O` output vec size
pub trait Layers<F, C, const I: usize, const O: usize> {

    /// go forward and get calculation result of all layers
    fn forward(&self, item: &SVector<f64,I>) -> C;

    /// backward and get gradient
    /// 
    /// `gradient`: Partial derivative of `E` with respect to `out`, where E is loss function
    fn backward(&self, input: &SVector<f64,I>, gradient: SVector<f64, O>, calc: C) -> Self;

    /// `rate`: learn rate
    fn update(&mut self, rate: f64, gradients: impl IntoIterator<Item = Self>);

    fn test(&self, item: &SVector<f64, I>) -> SVector<f64,O>;

}

pub trait Calculation<const O: usize> {
    fn out(&self) -> &SVector<f64,O>;
}

/// `S`: cur layer size
/// 
/// `P`: pre layer size
#[derive(Clone)]
pub struct Layer<const S: usize, const P: usize> {
    pub w: SMatrix<f64, S, P>,
    pub b: SVector<f64, S>,
}

impl<const S: usize, const P: usize> Layer<S, P> {

    #[inline]
    pub fn new() -> Self {
        Self {
            w: SMatrix::repeat(0f64),
            b: SVector::repeat(0f64), 
        }
    }

    #[inline]
    pub fn calc(&self, input: &SVector<f64,P>) -> SVector<f64,S> {
        self.w * input + self.b
    }

    pub fn random() -> Self {
        let iter = (1..).map(|_| rand::random::<f64>());
        Self {
            w: SMatrix::from_iterator(iter.clone()),
            b: SVector::from_iterator(iter),
        }
    }
}

impl<const S: usize, const P: usize> Default for Layer<S, P> {
    #[inline]
    fn default() -> Self {
        Self {
            w: SMatrix::repeat(0f64),
            b: SVector::repeat(0f64),
        }
    }
}

impl<const S: usize, const P: usize> AsRef<Self> for Layer<S, P> {
    #[inline]
    fn as_ref(&self) -> &Self {
        &self
    }
}

impl<T: AsRef<Self>,const S: usize, const P: usize> AddAssign<&T> for Layer<S, P>{
    #[inline]
    fn add_assign(&mut self, rhs: &T) {
        let rhs = rhs.as_ref();
        self.w += rhs.w;
        self.b += rhs.b;
    }
}

impl<const S: usize, const P: usize> SubAssign<(&SMatrix<f64, S, P>, &SVector<f64, S>)> for Layer<S, P>{
    fn sub_assign(&mut self, rhs: (&SMatrix<f64, S, P>, &SVector<f64, S>)) {
        self.w -= rhs.0;
        self.b -= rhs.1;
    }
}

impl<T: AsRef<Self>, const S: usize, const P: usize> SubAssign<&T> for Layer<S, P>{
    #[inline]
    fn sub_assign(&mut self, rhs: &T) {
        let rhs = rhs.as_ref();
        self.w -= rhs.w;
        self.b -= rhs.b;
    }
}

impl<const S: usize, const P: usize> DivAssign<f64> for Layer<S, P>{
    #[inline]
    fn div_assign(&mut self, rhs: f64) {
        self.w /= rhs;
        self.b /= rhs;
    }
}

impl<const S: usize, const P: usize> MulAssign<f64> for Layer<S, P>{
    #[inline]
    fn mul_assign(&mut self, rhs: f64) {
        self.w *= rhs;
        self.b *= rhs;
    }
}
