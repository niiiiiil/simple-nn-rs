#![allow(non_snake_case)]

use na::SVector;

pub trait ActivitionFunc {
    fn f(x: f64) -> f64;
    // fn d(x: f64) -> f64;
    fn d_from_y(y: f64) -> f64;

    fn fv<const S: usize>(v: &SVector<f64,S>) -> SVector<f64,S> {
        v.map(Self::f)
    }

    fn d_from_yv<const S: usize>(v: &SVector<f64,S>) -> SVector<f64,S> {
        v.map(Self::d_from_y)
    }
}

#[derive(Clone, Copy, Default,)]
pub struct Tanh;

impl ActivitionFunc for Tanh {
    #[inline]
    fn f(x: f64) -> f64 {
        let ex = x.exp();
        let e_x = (-x).exp();
        (ex - e_x)/(ex + e_x)
    }

    #[inline]
    fn d_from_y(y: f64) -> f64 {
        1f64 - y.powi(2)
    }
}

#[derive(Clone, Copy, Default,)]
pub struct Sigmoid;

impl ActivitionFunc for Sigmoid {
    #[inline]
    fn f(x: f64) -> f64 {
        1. / 1. + (-x).exp()
    }

    #[inline]
    fn d_from_y(y: f64) -> f64 {
        y * (1f64 - y)
    }
}

pub trait LossFunc {
    fn f<const S: usize>(Y: &SVector<f64,S>, y: &SVector<f64,S>) -> f64;
    fn d<const S: usize>(Y: &SVector<f64,S>, y: &SVector<f64,S>) -> SVector<f64,S>;
}

#[derive(Clone, Copy, Default,)]
pub struct CrossEntroy;

impl LossFunc for CrossEntroy {
    fn f<const S: usize>(Y: &SVector<f64,S>, y: &SVector<f64,S>) -> f64 {
        softmax(y)
            .iter()
            .zip(Y.iter())
            .fold(0., |pre, (p, Y)| {
                pre - Y * p.log2()
            })
    }

    fn d<const S: usize>(Y: &SVector<f64,S>, y: &SVector<f64,S>) -> SVector<f64,S> {
        let mut result = softmax(y);
        result.iter_mut()
            .zip(Y.iter())
            .for_each(|(p, Y)| {
                *p -= Y
            });
        result
    }
}

pub trait SoftMax<const S: usize> {
    fn softmax(&mut self);
}

impl<const S: usize> SoftMax<S> for SVector<f64, S>{
    #[inline]
    fn softmax(&mut self) {
        softmax_with(self)
    }
}

#[inline]
pub fn softmax<const S: usize>(x: &SVector<f64,S>) -> SVector<f64,S>  {
    let mut e = x.map(|val| val.exp());
    let sum = e.sum();
    e.iter_mut().for_each(|e| *e /= sum);
    e
}

#[inline]
pub fn softmax_with<const S: usize>(x: &mut SVector<f64,S>) {
    x.iter_mut().for_each(|val| *val = val.exp());
    let sum = x.sum();
    x.iter_mut().for_each(|e| *e /= sum);
}

pub struct DistanceFunc;

impl LossFunc for DistanceFunc {
    fn f<const S: usize>(Y: &SVector<f64,S>, y: &SVector<f64,S>) -> f64 {
        y.iter().zip(Y.iter()).fold(0f64, |pre, (y, Y)| {
            pre + (y-Y).powi(2)
        }) / 2f64
    }

    fn d<const S: usize>(Y: &SVector<f64,S>, y: &SVector<f64,S>) -> SVector<f64,S> {
        y.zip_map(&Y, |y, Y| y-Y)
    }
}
