use func::*;
use model::*;
use train::*;

pub use nn_macros::derive_layers;

pub mod func;
pub mod model;
mod train;

/// `F`: Activition function
/// 
/// `L`: Layers
/// 
/// `S`: input vector size
/// 
/// `E`: output vector size
#[derive(Clone)]
pub struct Network<A, C, L, Cal, const I: usize, const O: usize>
where
    L: Layers<A, Cal, I, O>
{
    layers: L,
    config: Config<A, C>,
    _maker: std::marker::PhantomData<Cal>
}

impl<A, C, L, Cal, const I: usize, const O: usize> Network<A, C, L, Cal, I, O>
where
    A: ActivitionFunc,
    C: LossFunc,
    L: Layers<A, Cal, I, O>,
    Cal: Calculation<O>,
{
    #[inline]
    pub fn cfg(layers: L, config: Config<A, C>) -> Network<A, C, L, Cal, I, O> {
        Network {
            layers,
            config,
            _maker: Default::default(),
        }
    }

    #[inline]
    pub fn train<'a, T>(self, data: T) -> Trainer<'a, A, C, L, Cal, I, O>
    where
        T: IntoIterator<Item = &'a self::Item<I, O>>,
        Self: 'a
    {
        Trainer::new(self.layers, data.into_iter(), self.config)
    }

}

impl<L, Cal, const I: usize, const O: usize> Network<Sigmoid, CrossEntroy, L, Cal, I, O>
where
    L: Layers<Sigmoid, Cal, I, O>,
    Cal: Calculation<O>
{
    /// use default [`Config`]:
    #[inline]
    pub fn new(layers: L) -> Network<Sigmoid, CrossEntroy, L, Cal, I, O> {
        Network::cfg(layers, Default::default())
    }
}

pub struct Item<const S: usize, const O: usize> {
    pub data: SVector<f64, S>,
    pub label: SVector<f64, O>,
}

/// default:
/// 
/// ```
/// learn_rate: 0.005
/// batch_size: 100
/// iter_num: 10_000
/// activition_func: Sigmoid
/// ```
#[derive(Clone)]
pub struct Config<A,C> {
    pub learn_rate: f64,
    pub batch_size: usize,
    pub iter_num: usize,
    pub actvt_func: A,
    pub loss_func: C,
}

impl<A: ActivitionFunc, C: LossFunc> Config<A, C> {
    pub fn default_with_func(actvt_func: A, cost_func: C) -> Config<A, C>{
        Self {
            learn_rate: 0.005,
            batch_size: 100,
            iter_num: 10_000,
            actvt_func,
            loss_func: cost_func
        }
    }
}

impl<A: Default + ActivitionFunc, C: Default + LossFunc> Default for Config<A, C> {
    fn default() -> Self {
        Config::default_with_func(A::default(), C::default())
    }
}
